pub mod mysql;
pub mod postgresql;
pub mod sqlite;

pub struct Migration {
    pub description: &'static str,
    pub statements: &'static [&'static str],
}
