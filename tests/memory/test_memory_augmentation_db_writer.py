from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from memori.memory.augmentation._db_writer import DbWriterRuntime, WriteTask


class TestWriteTask:
    def test_write_task_execution(self):
        mock_driver = Mock()
        mock_method = Mock(return_value="success")
        mock_driver.conversation.message.create = mock_method

        task = WriteTask(
            conn_factory=lambda: object(),
            method_path="conversation.message.create",
            args=(1, 2),
            kwargs={"key": "value"},
        )

        result = task.execute(mock_driver)

        mock_method.assert_called_once_with(1, 2, key="value")
        assert result == "success"


class TestDbWriterRuntime:
    def test_enqueue_write_success(self):
        import queue as queue_module

        runtime = DbWriterRuntime()
        runtime.queue = queue_module.Queue(maxsize=1000)
        task = WriteTask(
            conn_factory=lambda: object(),
            method_path="conversation.message.create",
        )

        result = runtime.enqueue_write(task, timeout=1.0)

        assert result is True
        assert runtime.queue.qsize() == 1

    def test_enqueue_write_full_queue(self):
        import queue as queue_module

        runtime = DbWriterRuntime()
        runtime.queue = Mock()
        runtime.queue.put = Mock(side_effect=queue_module.Full("Queue full"))
        runtime.queue.maxsize = 1000

        task = WriteTask(
            conn_factory=lambda: object(),
            method_path="conversation.message.create",
        )
        result = runtime.enqueue_write(task, timeout=1.0)

        assert result is False

    def test_drain_batches_opens_connection_only_when_work(self, mocker):
        import queue as queue_module

        runtime = DbWriterRuntime()
        runtime.queue = queue_module.Queue(maxsize=1000)
        runtime.batch_timeout = 0.01
        runtime.batch_size = 100

        driver = SimpleNamespace(
            conversation=SimpleNamespace(
                message=SimpleNamespace(create=mocker.Mock(return_value=None))
            )
        )
        adapter = SimpleNamespace(
            flush=mocker.Mock(),
            commit=mocker.Mock(),
            rollback=mocker.Mock(),
        )

        events: list[str] = []

        @contextmanager
        def fake_connection_context(_factory):
            events.append("enter")
            yield object(), adapter, driver
            events.append("exit")

        mocker.patch(
            "memori.memory.augmentation._db_writer.connection_context",
            fake_connection_context,
        )

        # First drain: no work -> no connection open.
        runtime._drain_batches()
        assert events == []

        def factory():
            return object()

        runtime.enqueue_write(
            WriteTask(conn_factory=factory, method_path="conversation.message.create")
        )
        runtime.enqueue_write(
            WriteTask(conn_factory=factory, method_path="conversation.message.create")
        )

        runtime._drain_batches()

        assert events == ["enter", "exit"]
        assert driver.conversation.message.create.call_count == 2
        adapter.flush.assert_called()
        adapter.commit.assert_called()
