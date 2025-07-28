from models import CallAnalysis
from collections import defaultdict
from datetime import datetime

def compute_agent_summary(agent=None, start=None, end=None):
    query = CallAnalysis.query

    # Apply filters if provided
    if agent:
        query = query.filter_by(agent=agent)
    if start:
        query = query.filter(CallAnalysis.created_at >= datetime.strptime(start, '%Y-%m-%d'))
    if end:
        query = query.filter(CallAnalysis.created_at <= datetime.strptime(end, '%Y-%m-%d'))

    records = query.all()

    weekly_data = []
    lead_distribution = {
        'High Interest': 0,
        'Moderate Interest': 0,
        'Low Interest': 0
    }
    high_interest_by_agent = defaultdict(int)

    all_agents = {r.agent for r in CallAnalysis.query.with_entities(CallAnalysis.agent).distinct()}
    for agent_name in all_agents:
        high_interest_by_agent[agent_name] = 0

    for rec in records:
        week = rec.created_at.strftime('%G-W%V')

        # ✅ Updated thresholds
        if rec.lead_score >= 50:
            category = 'High Interest'
        elif rec.lead_score >= 30:
            category = 'Moderate Interest'
        else:
            category = 'Low Interest'

        lead_distribution[category] += 1

        if category == 'High Interest':
            high_interest_by_agent[rec.agent] += 1

        week_entry = next((w for w in weekly_data if w['week'] == week), None)
        if not week_entry:
            week_entry = {
                'week': week,
                'High Interest': 0,
                'Moderate Interest': 0,
                'Low Interest': 0
            }
            weekly_data.append(week_entry)

        week_entry[category] += 1

    weekly_data.sort(key=lambda x: x['week'])

    return (
        {
            'weekly_leads': weekly_data,
            'lead_distribution': lead_distribution
        },
        dict(high_interest_by_agent)
    )
