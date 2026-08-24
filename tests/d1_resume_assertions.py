"""Exact nested checkpoint-state assertion shared by resume continuation tests."""

from __future__ import annotations

import torch


def assert_nested_equal(actual: object, expected: object) -> None:
    if type(expected) is torch.Tensor:
        assert type(actual) is torch.Tensor
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    elif type(expected) is dict:
        assert type(actual) is dict and set(actual) == set(expected)
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif type(expected) in (list, tuple):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected
