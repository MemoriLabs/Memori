pub mod mysql;
pub mod postgresql;
pub mod sqlite;

pub struct Migration {
    pub description: &'static str,
    /// One or more SQL statements that make up this migration step.
    pub statements: &'static [&'static str],
}
