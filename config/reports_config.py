# All report configs live here.
# Fields:
# - file:            path to the data file (Excel/CSV/PDF)
# - file_type:       "excel" | "pdf" (used to decide loading method)
# - description:     short text shown in sidebar
# - focus_areas:     list of strings (shown in sidebar)
# - prompt_file:     path to the external prompt text file (optional; fallback if missing/empty)
# - sample_questions:list of dicts: {"q": "...", "context": "..."} ; context is optional and may be empty

REPORTS_CONFIG = {
    "Weekly Merged Report": {
        "file": "Weekly Merged Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Customer self-serve, app adoption, and digital engagement metrics.",
            "Operational efficiency, customer satisfaction, and workload management for inbound contact centers",
            "Direct (Call Center) and Online sales channels for both Wireless and Wireline products"
        ],
        "prompt_file": "prompts/Weekly Merged Prompt.txt",
        "sample_questions": [
            {"q": "What is the overall story of this month?"},
            {"q": "Why is Sales SLA so much higher than Service SLA?"},
            {"q": "Why is YoY call propensity going up in this view when overall call propensity is going down YoY?"},
            {"q": "If outages are driving SLA pressures this month, what is the SLA if we take out the 3 days that were impacted by an outage?"}
        ],
        #confluence page
        "supplements": [
            {
                "topic": "CCTS",
                "keywords": [
                    "ccts", "complaint", "complaints","commission for complaints"],
                "file": "confluence.xlsx",   # confluence Excel
                "file_type": "excel",
                "sheet": "WE",                   
                "max_rows": 100,                    
                "window_weeks": 13                   
            }
]

    },

    "Weekly Overall Sales Report": {
        "file": "Weekly Overall Sales Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Strategic Overview – Key priorities and initiatives.",
            "Performance Metrics – Cross-functional KPIs and comparisons.",
            "Market Analysis – Trends, competition, opportunities.",
            "Operational Insights – Processes and efficiency.",
            "Future Roadmap – Upcoming projects and growth paths."
        ],
        "prompt_file": "prompts/Weekly Overall Sales Prompt.txt",
        "sample_questions": [
            {"q": "What are the key strategic priorities highlighted in the deck?"},
            {"q": "Which business areas showed the strongest performance metrics?"},
            {"q": "What market opportunities and competitive advantages are highlighted?"}
        ]
    },

    "Weekly Digital Inights Report": {
        "file": "Weekly Digital Inights Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Strategic Overview – Key priorities and initiatives.",
            "Performance Metrics – Cross-functional KPIs and comparisons.",
            "Market Analysis – Trends, competition, opportunities.",
            "Operational Insights – Processes and efficiency.",
            "Future Roadmap – Upcoming projects and growth paths."
        ],
        "prompt_file": "prompts/Weekly Digital Insights Prompt.txt",
        "sample_questions": [
            {"q": "What are the key strategic priorities highlighted in the deck?"},
            {"q": "Which business areas showed the strongest performance metrics?"},
            {"q": "What market opportunities and competitive advantages are highlighted?"}
        ]
    },

    "Weekly ASG Report": {
        "file": "Weekly ASG Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Strategic Overview – Key priorities and initiatives.",
            "Performance Metrics – Cross-functional KPIs and comparisons.",
            "Market Analysis – Trends, competition, opportunities.",
            "Operational Insights – Processes and efficiency.",
            "Future Roadmap – Upcoming projects and growth paths."
        ],
        "prompt_file": "prompts/Weekly ASG Prompt.txt",
        "sample_questions": [
            {"q": "What are the key strategic priorities highlighted in the deck?"},
            {"q": "Which business areas showed the strongest performance metrics?"},
            {"q": "What market opportunities and competitive advantages are highlighted?"}
        ]
    },

    "Weekly Fields Report": {
        "file": "Weekly Fields Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Strategic Overview – Key priorities and initiatives.",
            "Performance Metrics – Cross-functional KPIs and comparisons.",
            "Market Analysis – Trends, competition, opportunities.",
            "Operational Insights – Processes and efficiency.",
            "Future Roadmap – Upcoming projects and growth paths."
        ],
        "prompt_file": "prompts/Weekly Fields Prompt.txt",
        "sample_questions": [
            {"q": "What are the key strategic priorities highlighted in the deck?"},
            {"q": "Which business areas showed the strongest performance metrics?"},
            {"q": "What market opportunities and competitive advantages are highlighted?"}
        ]
    },
    
    "Weekly Pres Deck Report": {
        "file": "Weekly Pres Deck Report.pdf",
        "file_type": "pdf",
        "description": "Comprehensive deck with strategic insights, performance metrics, and recommendations.",
        "focus_areas": [
            "Strategic Overview – Key priorities and initiatives.",
            "Performance Metrics – Cross-functional KPIs and comparisons.",
            "Market Analysis – Trends, competition, opportunities.",
            "Operational Insights – Processes and efficiency.",
            "Future Roadmap – Upcoming projects and growth paths."
        ],
        "prompt_file": "prompts/Weekly Pres Deck Prompt.txt",
        "sample_questions": [
            {"q": "What are the key strategic priorities highlighted in the deck?"},
            {"q": "Which business areas showed the strongest performance metrics?"},
            {"q": "What market opportunities and competitive advantages are highlighted?"}
        ]
    },


    "Digital Sales Report": {
        "file": "Digital_Sales_Report.xlsx",
        "file_type": "excel",
        "description": "Weekly sales and traffic report analyzing gross sales trends, customer segmentation, and channel performance.",
        "focus_areas": [
            "Gross Sales Trends – Compare week-over-week sales fluctuations. Identify key drivers: new vs existing customers, Mobility vs Residential sales.",
            "Overall Shop Traffic – The number of total visits on Shop pages.",
            "Overall Channel Mix – Evaluate the proportion of sales driven by online channels compared to other sales avenues.",
            "Close Rate – Measure the close rate by calculating sales divided by total shop traffic."
        ],
        "prompt_file": "prompts/Weekly Digital Sales Prompt.txt",
        "sample_questions": [
            {"q": "What is the Activation% for the most recent week?", "context": "Analyze the Activation% row vs previous periods."},
            {"q": "Which line of business fluctuated more in the latest week, Mobility or Residential?"},
            {"q": "How did close rates (vs Shop Traffic) change week-over-week?"}
        ]
    },
}
