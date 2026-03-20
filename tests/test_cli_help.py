from space_debris_rl.cli import main


def test_cli_help_runs(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    out = capsys.readouterr().out
    assert "space-debris-rl" in out
