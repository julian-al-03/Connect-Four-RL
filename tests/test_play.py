from connect4.play import play_against


def scripted_input(monkeypatch, moves):
    moves = iter(moves)
    monkeypatch.setattr("builtins.input", lambda _: next(moves))


def scripted_agent(moves):
    moves = iter(moves)
    return lambda state: next(moves)


def test_human_wins_as_first_player(monkeypatch):
    scripted_input(monkeypatch, ["0", "0", "0", "0"])
    assert play_against(scripted_agent([1, 1, 1]), human_first=True) == 1


def test_human_loses_as_second_player(monkeypatch):
    scripted_input(monkeypatch, ["1", "1", "1"])
    assert play_against(scripted_agent([0, 0, 0, 0]), human_first=False) == -1


def test_invalid_input_is_asked_again(monkeypatch, capsys):
    scripted_input(monkeypatch, ["x", "7", "-1", "0", "0", "0", "0"])
    assert play_against(scripted_agent([1, 1, 1]), human_first=True) == 1
    assert capsys.readouterr().out.count("not a legal move") == 3


def test_full_column_is_asked_again(monkeypatch, capsys):
    # both players fill column 0, so the human's next move there must be refused
    scripted_input(monkeypatch, ["0", "0", "0", "0", "1", "1", "1", "1"])
    assert play_against(scripted_agent([0, 0, 0, 2, 2, 2]), human_first=True) == 1
    assert capsys.readouterr().out.count("not a legal move") == 1
