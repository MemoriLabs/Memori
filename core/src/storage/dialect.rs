#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dialect {
    Sqlite,
    Postgresql,
    Cockroachdb,
    Mysql,
}

impl std::str::FromStr for Dialect {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sqlite" => Ok(Self::Sqlite),
            "postgresql" => Ok(Self::Postgresql),
            "cockroachdb" => Ok(Self::Cockroachdb),
            "mysql" => Ok(Self::Mysql),
            other => Err(format!("unsupported dialect: {other}")),
        }
    }
}

impl Dialect {
    /// PostgreSQL (and CockroachDB) leave transactions open on error and need
    /// an explicit ROLLBACK before the connection can be reused. SQLite cleans
    /// up automatically; MySQL also requires explicit rollback.
    pub fn requires_rollback_on_error(&self) -> bool {
        matches!(
            self,
            Dialect::Postgresql | Dialect::Cockroachdb | Dialect::Mysql
        )
    }
}

impl std::fmt::Display for Dialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Sqlite => "sqlite",
            Self::Postgresql => "postgresql",
            Self::Cockroachdb => "cockroachdb",
            Self::Mysql => "mysql",
        };
        write!(f, "{s}")
    }
}
