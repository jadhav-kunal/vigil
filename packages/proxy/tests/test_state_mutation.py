"""Word-boundary verb matching for state-mutation inference (spec 4.2)."""

from vigil_proxy.state_mutation import caused_state_mutation


def test_mutating_verbs_match():
    for name in ("write_file", "create_user", "db_insert", "deleteRecord", "send-email"):
        assert caused_state_mutation(name) is True, name


def test_read_only_tools_do_not_match():
    for name in ("get_status", "read_file", "list_items", "search_web", "fetch_page"):
        assert caused_state_mutation(name) is False, name


def test_substring_in_is_not_a_false_match():
    # The classic trap: 'insert'/'in' must not be matched inside 'find' / 'inspect'.
    assert caused_state_mutation("find") is False
    assert caused_state_mutation("inspect") is False
    assert caused_state_mutation("inventory") is False


def test_metadata_override_wins():
    assert caused_state_mutation("read_file", metadata_override=True) is True
    assert caused_state_mutation("write_file", metadata_override=False) is False


def test_no_tool_is_not_a_mutation():
    assert caused_state_mutation(None) is False
    assert caused_state_mutation("") is False


def test_camel_case_tokenization():
    assert caused_state_mutation("updateUserProfile") is True
    assert caused_state_mutation("getUserProfile") is False
