import pytest

from hd.humanizer.facts import check_facts, names, number_values


def test_numbers_parse_digits_and_spelled_out_forms_to_the_same_value():
    assert number_values("She was seventy-eight, 78 years old, or seventy eight.") == {78}
    assert number_values("three hundred and twelve deaths, 312 in all") == {312}
    assert number_values("1,200 stones, forty-seven years, 45 men") == {1200, 47, 45}
    assert number_values("two hundred and five, then one thousand two hundred") == {205, 1200}


def test_a_mangled_number_parses_to_a_different_value():
    assert number_values("Seven-eight years old") == {15}  # not 78: this is the mistake the guard exists to catch


def test_pronoun_one_and_hyperbole_are_not_facts():
    assert number_values("No one came. The one she loved. One of them.") == set()
    assert number_values("a hundred unspoken confessions, a thousand years, a million reasons") == set()
    assert number_values("100 men, three wars") == {100, 3}  # digits and specific counts still count


def test_names_are_capitalized_words_that_are_not_sentence_initial():
    text = 'Maria climbed. "Esther said Spokane was far," said Marta, and I agreed. Tuesday came; then John\'s dog.'
    assert names(text) == {"spokane", "marta", "john"}  # Maria/Esther/Tuesday start a sentence or quote: a known blind spot
    assert names("It was Spokane in the rain. She met Dr. Okafor at Mercy Hospital.") >= {"spokane", "okafor", "mercy", "hospital"}


def test_check_facts_flags_changed_numbers_and_missing_names_only():
    original = "At seventy-eight, she walked through Spokane with Marta. Three hundred and twelve people had died."
    assert check_facts(original, "She was 78. Marta and she crossed Spokane; 312 had died.").missing_numbers == []  # reworded, same facts
    mangled = check_facts(original, "Seven-eight years old, she walked through Spokane with Marta. 312 had died.")
    assert mangled.missing_numbers == [78]
    lost = check_facts(original, "At seventy-eight, she walked through the city with her friend. 312 died.")
    assert lost.missing_names == ["marta", "spokane"] and lost.missing_name_frac == 1.0 and lost.missing_numbers == []
    assert check_facts(original, original.upper()).missing_names == []  # presence is case-insensitive


def test_text_without_names_or_numbers_never_flags():
    r = check_facts("The cat sat on the mat. It was late.", "Late, the cat sat down.")
    assert r.missing_numbers == [] and r.missing_names == [] and r.missing_name_frac == 0.0
