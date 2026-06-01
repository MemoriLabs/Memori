use crate::storage::connection::StorageConnection;
use crate::storage::dialect::Dialect;
use crate::storage::drivers::{mysql, postgresql, sqlite};
use crate::storage::migrations;
use crate::storage::models::HostStorageError;

/// Returns true only when the error means the schema_version table does not exist yet.
/// Any other error — permission denied, transient failure, corrupted schema — should
/// propagate rather than being silently treated as a fresh database.
fn is_table_not_found(dialect: &Dialect, err: &HostStorageError) -> bool {
    match dialect {
        // PostgreSQL SQLSTATE 42P01 = undefined_table
        Dialect::Postgresql | Dialect::Cockroachdb => err.code == "42P01",
        // MySQL/mysql2 SQLSTATE 42S02 = base table or view not found
        Dialect::Mysql => err.code == "42S02",
        // SQLite has no SQLSTATE; better-sqlite3 sets code="SQLITE_ERROR" and embeds
        // "no such table" in the message for queries against a missing table.
        Dialect::Sqlite => err.message.contains("no such table"),
    }
}

/// Runs pending database migrations to bring the schema to the latest version.
///
/// Each migration batch runs in its own transaction so a partial failure only
/// rolls back that batch, not the entire migration history — matching the
/// behaviour of the TS Builder exactly.
pub fn run(conn: &dyn StorageConnection, dialect: &Dialect) -> Result<(), HostStorageError> {
    let current_version = read_schema_version(conn, dialect)?;

    log::info!("[Memori] Currently at schema revision #{current_version}.");

    let migrations = get_migrations(dialect);
    let max_version = migrations.iter().map(|(v, _)| *v).max().unwrap_or(0);

    if current_version == max_version as i64 {
        log::info!("[Memori] Data structures are up-to-date.");
        return Ok(());
    }

    let mut num = current_version;
    loop {
        num += 1;
        let batch = match migrations.iter().find(|(v, _)| *v == num as u32) {
            Some((_, batch)) => *batch,
            None => break,
        };

        log::info!("[Memori] Building revision #{num}...");

        for migration in batch.iter() {
            log::info!("  -> {}", migration.description);
            conn.begin()?;
            let result: Result<(), HostStorageError> = (|| {
                for statement in migration.statements.iter() {
                    conn.execute(statement, vec![])?;
                }
                conn.commit()
            })();
            if let Err(e) = result {
                let _ = conn.rollback();
                return Err(e);
            }
        }
    }

    // num is one past the last applied migration after breaking, subtract 1.
    conn.begin()?;
    let version_result: Result<(), HostStorageError> = (|| {
        delete_schema_version(conn, dialect)?;
        write_schema_version(conn, dialect, num - 1)?;
        conn.commit()
    })();
    if let Err(e) = version_result {
        let _ = conn.rollback();
        return Err(e);
    }

    log::info!("[Memori] Build executed successfully!");
    Ok(())
}

fn read_schema_version(
    conn: &dyn StorageConnection,
    dialect: &Dialect,
) -> Result<i64, HostStorageError> {
    let result = match dialect {
        Dialect::Sqlite => sqlite::schema_version_read(conn),
        Dialect::Postgresql | Dialect::Cockroachdb => postgresql::schema_version_read(conn),
        Dialect::Mysql => mysql::schema_version_read(conn),
    };

    match result {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Ok(0),
        Err(e) => {
            // Always rollback for dialects that need it — a failed query may leave an
            // implicit transaction open (PostgreSQL, CockroachDB, MySQL).
            if dialect.requires_rollback_on_error() {
                let _ = conn.rollback();
            }
            if is_table_not_found(dialect, &e) {
                // schema_version table doesn't exist yet — first run, start from version 0.
                Ok(0)
            } else {
                // Permission error, transient failure, or corrupted schema — propagate rather
                // than treating an unknown DB state as a fresh install.
                Err(e)
            }
        }
    }
}

fn delete_schema_version(
    conn: &dyn StorageConnection,
    dialect: &Dialect,
) -> Result<(), HostStorageError> {
    match dialect {
        Dialect::Sqlite => sqlite::schema_version_delete(conn),
        Dialect::Postgresql | Dialect::Cockroachdb => postgresql::schema_version_delete(conn),
        Dialect::Mysql => mysql::schema_version_delete(conn),
    }
}

fn write_schema_version(
    conn: &dyn StorageConnection,
    dialect: &Dialect,
    num: i64,
) -> Result<(), HostStorageError> {
    match dialect {
        Dialect::Sqlite => sqlite::schema_version_create(conn, num),
        Dialect::Postgresql | Dialect::Cockroachdb => postgresql::schema_version_create(conn, num),
        Dialect::Mysql => mysql::schema_version_create(conn, num),
    }
}

fn get_migrations(dialect: &Dialect) -> &'static [(u32, &'static [migrations::Migration])] {
    match dialect {
        Dialect::Sqlite => migrations::sqlite::MIGRATIONS,
        Dialect::Postgresql | Dialect::Cockroachdb => migrations::postgresql::MIGRATIONS,
        Dialect::Mysql => migrations::mysql::MIGRATIONS,
    }
}
