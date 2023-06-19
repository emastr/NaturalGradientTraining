import pytest

from natgrad.domains import Interval

@pytest.mark.integration
def test_Interval():
    a = 0.
    b = 1.
    interval = Interval(a, b)
    assert interval._a == 0.0

def test_Interval2():
    a = 0.
    b = 1.
    interval = Interval(a, b)
    assert interval._a == 0.0
   