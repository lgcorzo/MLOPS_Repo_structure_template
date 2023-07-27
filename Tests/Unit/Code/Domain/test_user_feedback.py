from Code.Domain.Models.user_feedback import Feedback


def test_user_feedback_init():
    feedback = Feedback("example_id", "Everything correct", True)
    assert feedback.id == 'example_id'
    assert feedback.comments == 'Everything correct'
    assert feedback.validation


def test_user_feedback_dict():
    feedback = Feedback("example_id", "Everything correct", True)
    dictio = feedback.as_dict()
    assert dictio['id'] == 'example_id'
    assert dictio['comments'] == 'Everything correct'
    assert dictio['validation']
