import json
from datetime import datetime
from typing import Dict, Any

class ReportError(Exception):
    pass

def build_report(source: str, primary_emotion: str, probabilities: Dict[str, float], scores: Dict[str, float]) -> Dict[str, Any]:
    report = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'source': source,
        'primary_emotion': primary_emotion,
        'probabilities': {k: float(v) for k, v in probabilities.items()},
        'scores': {k: float(v) for k, v in scores.items()}
    }
    return report

def serialize_report_json(report: Dict[str, Any]) -> bytes:
    try:
        return json.dumps(report, indent=2).encode('utf-8')
    except Exception as e:
        raise ReportError(f'Could not serialize report to JSON: {e}')
