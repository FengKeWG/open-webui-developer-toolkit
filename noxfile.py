import nox

nox.options.sessions = ["lint", "tests"]

@nox.session(venv_backend="none")
def lint(session):
    session.run("ruff", "check", "functions", "tools", ".tests", ".scripts")

@nox.session(venv_backend="none")
def tests(session):
    import os

    env = session.env.copy()
    project_root = os.getcwd()
    env["PYTHONPATH"] = f"{project_root}/src" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    session.run(
        "pytest",
        "-vv",
        "--cov=functions",
        "--cov-report=term-missing",
        *session.posargs,
        env=env,
    )
