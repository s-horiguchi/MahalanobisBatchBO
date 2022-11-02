import os

WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
if WEBHOOK_URL:
    try:
        import slackweb
    except ImportError:
        print("Failed to import slackweb")

        def notify(text: str) -> None:
            print(text)
    else:
        slack = slackweb.Slack(url=WEBHOOK_URL)

        def notify(text: str) -> None:
            print(text)
            slack.notify(text=text)
else:
    print("Environment variable WEBHOOK_URL is not set")

    def notify(text: str) -> None:
        print(text)
