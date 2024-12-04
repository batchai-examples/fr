import pytest

def test_pytest_addoption():
    """
    Test that custom command-line options are added to pytest.
    """
    # Arrange
    parser = pytest.config.Parser()

    # Act
    pytest_addoption(parser)

    # Assert
    assert "--slow" in parser.get_option("--slow")

def test_pytest_collection_modifyitems_with_slow_option():
    """
    Test that slow tests are not removed if the --slow option is provided.
    """
    # Arrange
    config = pytest.config.Config()
    items = [pytest.Item("tests/test_example.py::test_example", keywords={"slow": True})]
    config._args = ["--slow"]

    # Act
    pytest_collection_modifyitems(config, items)

    # Assert
    assert len(items) == 1

def test_pytest_collection_modifyitems_without_slow_option():
    """
    Test that slow tests are removed if the --slow option is not provided.
    """
    # Arrange
    config = pytest.config.Config()
    items = [pytest.Item("tests/test_example.py::test_example", keywords={"slow": True})]
    config._args = []

    # Act
    pytest_collection_modifyitems(config, items)

    # Assert
    assert len(items) == 0

def test_pytest_sessionstart():
    """
    Test that session configurations are initialized for pytest.
    """
    # Arrange
    session = pytest.Session()

    # Act
    pytest_sessionstart(session)

    # Assert
    assert TMP.exists()
    assert not list(TMP.iterdir())

def test_pytest_terminal_summary():
    """
    Test that cleanup operations are performed after pytest session.
    """
    # Arrange
    terminalreporter = pytest.terminal.TerminalReporter()
    exitstatus = 0
    config = pytest.config.Config()

    # Act
    pytest_terminal_summary(terminalreporter, exitstatus, config)

    # Assert
    assert not list(Path("weights").rglob("*.onnx"))
    assert not list(Path("weights").rglob("*.torchscript"))
    assert not Path("bus.jpg").exists()
    assert not Path("yolov8n.onnx").exists()
    assert not Path("yolov8n.torchscript").exists()
    assert not list(TMP.parents[1] / ".pytest_cache".iterdir())
    assert not TMP.exists()
