import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.example_atlas import REQUIRED, render_atlas, select_example


def rows():
    return [dict(split='train', instance_key='PRIVATE_ID', representation=r,
                 text='<SYNTHETIC> value=1', target='PRIVATE_TARGET', prediction_time='PRIVATE_TIME')
            for r in sorted(REQUIRED)]


def test_atlas_uses_saved_text_escapes_and_does_not_print_identifiers():
    chosen = select_example(rows())
    rendered = render_atlas({'synthetic': chosen}, 'commit')
    assert '&lt;SYNTHETIC&gt; value=1' in rendered
    assert all(secret not in rendered for secret in ('PRIVATE_ID', 'PRIVATE_TARGET', 'PRIVATE_TIME'))
    assert select_example(list(reversed(rows()))) == chosen


def test_atlas_fails_closed_on_test_or_missing_condition():
    forbidden = rows()
    forbidden[0]['split'] = 'test'
    with pytest.raises(ContractError):
        select_example(forbidden)
    with pytest.raises(ContractError):
        select_example(rows()[:-1])
