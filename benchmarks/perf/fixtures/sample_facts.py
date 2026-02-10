# 1,000 synthetic user facts as semantic triples (subject, predicate, object)
# Copy/paste into your project. Produces: user_data = {"user": "John", "facts": [(s,p,o), ...]}


def build_user_data() -> dict:
    facts: list[tuple[str, str, str]] = []

    def add(s: str, p: str, o: str) -> None:
        facts.append((s, p, o))

    # -----------------------------
    # Core profile facts (diverse)
    # -----------------------------
    add("John", "type", "Person")
    add("John", "has_given_name", "John")
    add("John", "prefers_language", "English")
    add("John", "prefers_time_format", "12-hour")
    add("John", "prefers_temperature_unit", "Fahrenheit")
    add("John", "prefers_distance_unit", "miles")
    add("John", "has_home_city", "St_Louis")
    add("John", "has_home_state", "Missouri")
    add("John", "has_home_country", "United_States")
    add("John", "has_role", "Software_Engineer")
    add("John", "works_in", "Technology")
    add("John", "interested_in", "AI")
    add("John", "interested_in", "cloud_infrastructure")
    add("John", "interested_in", "personal_finance")
    add("John", "interested_in", "fitness")
    add("John", "interested_in", "cooking")
    add("John", "interested_in", "travel")
    add("John", "interested_in", "productivity")
    add("John", "interested_in", "home_improvement")
    add("John", "uses_llm_for", "writing_assistance")
    add("John", "uses_llm_for", "coding_help")
    add("John", "uses_llm_for", "travel_planning")
    add("John", "uses_llm_for", "learning_new_topics")
    add("John", "uses_llm_for", "recipe_ideas")
    add("John", "uses_llm_for", "career_advice")
    add("John", "uses_llm_for", "data_analysis")
    add("John", "uses_llm_for", "brainstorming")
    add("John", "prefers_response_style", "structured")
    add("John", "prefers_response_style", "actionable")
    add("John", "concerned_about", "latency")
    add("John", "concerned_about", "cost")
    add("John", "concerned_about", "privacy")
    add("John", "primary_os", "macOS")
    add("John", "primary_browser", "Chrome")
    add("John", "primary_editor", "VS_Code")
    add("John", "primary_shell", "zsh")
    add("John", "primary_email_provider", "Gmail")
    add("John", "primary_calendar", "Google_Calendar")
    add("John", "primary_messaging_app", "Slack")
    add("John", "primary_code_host", "GitHub")
    add("John", "prefers_document_format", "Markdown")
    add("John", "prefers_spreadsheet_tool", "Google_Sheets")
    add("John", "uses_password_manager", "1Password")
    add("John", "uses_cloud_storage", "Google_Drive")
    add("John", "uses_issue_tracker", "GitHub_Issues")
    add("John", "uses_ci_cd", "GitHub_Actions")
    add("John", "prefers_container_runtime", "Docker")
    add("John", "prefers_database", "PostgreSQL")
    add("John", "prefers_cache", "Redis")
    add("John", "prefers_language_for_backend", "Python")
    add("John", "knows_language", "JavaScript")
    add("John", "knows_language", "SQL")
    add("John", "knows_language", "Bash")
    add("John", "learning_language", "Rust")
    add("John", "prefers_testing_framework", "pytest")
    add("John", "prefers_package_manager", "uv")
    add("John", "uses_llm_provider", "OpenAI")
    add("John", "uses_llm_provider", "Anthropic")
    add("John", "uses_model_family", "GPT")
    add("John", "uses_model_family", "Claude")
    add("John", "has_hobby", "fantasy_football")
    add("John", "follows_sport", "NFL")
    add("John", "follows_sport", "NHL")
    add("John", "prefers_coffee_drink", "latte")
    add("John", "prefers_breakfast", "oatmeal")
    add("John", "prefers_lunch", "salad")
    add("John", "prefers_dinner", "grilled_chicken")
    add("John", "commutes_by", "car")
    add("John", "prefers_meeting_platform", "Zoom")

    # -----------------------------
    # Family / household facts
    # -----------------------------
    family = [
        ("Spouse_01", "spouse"),
        ("Child_01", "child"),
        ("Child_02", "child"),
        ("Parent_01", "parent"),
        ("Parent_02", "parent"),
        ("Sibling_01", "sibling"),
        ("Sibling_02", "sibling"),
        ("Pet_01", "pet"),
    ]
    for i, (entity, rel) in enumerate(family, start=1):
        add(entity, "type", "Person" if "Pet" not in entity else "Animal")
        add("John", f"has_{rel}", entity)
        add(entity, "has_first_name", entity.split("_")[0])
        add(entity, "located_in", f"City_{(i % 25) + 1:03d}")
        add(
            entity,
            "preferred_contact_method",
            ["text", "call", "email"][i % 3] if "Pet" not in entity else "n/a",
        )

    # -----------------------------
    # Friends (varied interests)
    # -----------------------------
    friend_interests = [
        "music",
        "travel",
        "tech",
        "food",
        "sports",
        "finance",
        "fitness",
        "gaming",
        "books",
        "photography",
    ]
    contact_methods = ["email", "text", "slack", "signal", "call"]
    for i in range(1, 61):  # 60 friends
        f = f"Friend_{i:03d}"
        add(f, "type", "Person")
        add("John", "has_friend", f)
        add(f, "located_in", f"City_{(i % 80) + 1:03d}")
        add(f, "interested_in", friend_interests[i % len(friend_interests)])
        add(f, "preferred_contact_method", contact_methods[i % len(contact_methods)])

    # -----------------------------
    # Coworkers (teams, roles, tools)
    # -----------------------------
    roles = [
        "Backend_Engineer",
        "Frontend_Engineer",
        "DevOps_Engineer",
        "Product_Manager",
        "Designer",
        "Data_Engineer",
    ]
    teams = ["Platform", "Infra", "Product", "Data", "Security", "Growth"]
    tools = ["Jira", "Linear", "Confluence", "Notion", "Figma", "Datadog", "Grafana"]
    for i in range(1, 41):  # 40 coworkers
        c = f"Coworker_{i:03d}"
        add(c, "type", "Person")
        add("John", "works_with", c)
        add(c, "has_role", roles[i % len(roles)])
        add(c, "member_of_team", teams[i % len(teams)])
        add(c, "uses_tool", tools[i % len(tools)])

    # -----------------------------
    # Places: cities, venues, travel
    # -----------------------------
    for i in range(1, 81):  # 80 cities
        city = f"City_{i:03d}"
        add(city, "type", "City")
        add(city, "in_country", "United_States" if i <= 60 else "International")
        add("John", "has_visited", city if i <= 45 else f"Planned_{city}")
        add(
            city,
            "has_timezone",
            "America/Chicago"
            if i % 3 == 0
            else "America/New_York"
            if i % 3 == 1
            else "America/Los_Angeles",
        )

    venue_types = [
        "Restaurant",
        "Coffee_Shop",
        "Gym",
        "Airport",
        "Hotel",
        "Park",
        "Museum",
        "Stadium",
    ]
    for i in range(1, 81):  # 80 venues
        v = f"Venue_{i:03d}"
        add(v, "type", venue_types[i % len(venue_types)])
        add(v, "located_in", f"City_{(i % 80) + 1:03d}")
        add("John", "likes_place", v if i % 4 != 0 else f"Neutral_{v}")
        add(v, "has_price_tier", ["$", "$$", "$$$"][i % 3])

    # -----------------------------
    # Devices, apps, services, accounts
    # -----------------------------
    devices = [
        "MacBook_Pro",
        "iPhone",
        "iPad",
        "AirPods",
        "Smart_TV",
        "Router",
        "NAS",
        "Mechanical_Keyboard",
        "Gaming_PC",
        "Monitor_34inch",
        "Standing_Desk",
        "Ergonomic_Chair",
        "Fitness_Tracker",
        "Smart_Speaker",
        "Kindle",
        "External_SSD",
        "Webcam",
        "Microphone",
        "Printer",
        "Smart_Thermostat",
    ]
    for d in devices:
        add(d, "type", "Device")
        add("John", "owns_device", d)
        add(d, "used_for", "work" if "MacBook" in d or "Monitor" in d else "personal")
        add(d, "has_status", "active")

    apps = [
        "Notion",
        "Slack",
        "Gmail",
        "Google_Calendar",
        "Spotify",
        "YouTube",
        "Netflix",
        "Amazon",
        "GitHub",
        "Figma",
        "Jira",
        "Linear",
        "Zoom",
        "1Password",
        "Google_Drive",
    ]
    for i, app in enumerate(apps, start=1):
        add(app, "type", "App")
        add("John", "uses_app", app)
        add(app, "has_version", f"{(i % 5) + 1}.{(i % 10)}.{(i % 20)}")

    return {"user": "John", "facts": facts}
