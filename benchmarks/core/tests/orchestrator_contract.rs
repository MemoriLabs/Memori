use engine_orchestrator::{EngineOrchestrator, OrchestratorError};

fn load_orchestrator_or_skip() -> Option<EngineOrchestrator> {
    match EngineOrchestrator::new(None) {
        Ok(orchestrator) => Some(orchestrator),
        Err(OrchestratorError::ModelError(message)) => {
            eprintln!("skipping test due to unavailable model runtime: {message}");
            None
        }
        Err(other) => panic!("Failed to initialize engine: {other}"),
    }
}

#[test]
fn executes_ping_command() {
    let Some(orchestrator) = load_orchestrator_or_skip() else {
        return;
    };
    let output = orchestrator.execute("ping");
    assert_eq!(output, Ok("pong".to_string()));
}

#[test]
fn maps_empty_command_to_invalid_input() {
    let Some(orchestrator) = load_orchestrator_or_skip() else {
        return;
    };
    let output = orchestrator.execute("");
    assert_eq!(
        output,
        Err(OrchestratorError::InvalidInput(
            "command cannot be empty".to_string()
        ))
    );
}

#[test]
fn returns_hello_world_message() {
    let Some(orchestrator) = load_orchestrator_or_skip() else {
        return;
    };
    let output = orchestrator.hello_world();
    assert_eq!(output, "hello world".to_string());
}

#[test]
fn accepts_postprocess_request_fire_and_forget() {
    let Some(orchestrator) = load_orchestrator_or_skip() else {
        return;
    };
    let accepted = orchestrator.postprocess_request("payload-1");
    assert!(accepted.is_ok());
    let accepted = accepted.unwrap();
    assert!(accepted.job_id > 0);
}

#[test]
fn rejects_empty_postprocess_payload() {
    let Some(orchestrator) = load_orchestrator_or_skip() else {
        return;
    };
    let output = orchestrator.postprocess_request("   ");
    assert_eq!(
        output,
        Err(OrchestratorError::InvalidInput(
            "postprocess payload cannot be empty".to_string()
        ))
    );
}
