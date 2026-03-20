def test_package_imports():
    import space_debris_rl

    assert isinstance(space_debris_rl.__version__, str)
