import json
from utilities.reporting import build_report, serialize_report_json

def test_build_and_serialize_report():
    probs = {'Happy': 0.8, 'Neutral': 0.2}
    scores = {'Anxiety': 10.0, 'Depression': 5.0, 'Stress': 7.5}
    rpt = build_report(source='test', primary_emotion='Happy', probabilities=probs, scores=scores)
    assert 'timestamp' in rpt
    assert rpt['source'] == 'test'
    assert rpt['primary_emotion'] == 'Happy'
    assert isinstance(rpt['probabilities']['Happy'], float)

    jb = serialize_report_json(rpt)
    parsed = json.loads(jb.decode('utf-8'))
    assert parsed['primary_emotion'] == 'Happy'
    assert abs(parsed['probabilities']['Happy'] - 0.8) < 1e-6
