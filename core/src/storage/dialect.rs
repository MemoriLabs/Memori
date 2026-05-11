#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dialect {
    Sqlite,
    Postgresql,
    Cockroachdb,
    Mysql,
}

impl Dialect {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "sqlite" => Some(Self::Sqlite),
            "postgresql" => Some(Self::Postgresql),
            "cockroachdb" => Some(Self::Cockroachdb),
            "mysql" => Some(Self::Mysql),
            _ => None,
        }
    }

    /// PostgreSQL (and CockroachDB) leave transactions open on error and need
    /// an explicit ROLLBACK before the connection can be reused. SQLite cleans
    /// up automatically; MySQL also requires explicit rollback.
    pub fn requires_rollback_on_error(&self) -> bool {
        matches!(
            self,
            Dialect::Postgresql | Dialect::Cockroachdb | Dialect::Mysql
        )
    }

    pub fn is_cockroachdb(&self) -> bool {
        matches!(self, Dialect::Cockroachdb)
    }
}
