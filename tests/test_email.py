import base64

import pytest

from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.utils import build_email_message, send_email


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO9/rfUAAAAASUVORK5CYII="
)


@pytest.fixture
def papers() -> list[Paper]:
    return [
        Paper(
            source="arxiv",
            title="Test Paper",
            authors=["Test Author", "Test Author 2"],
            abstract="Test Abstract",
            url="https://arxiv.org/abs/2512.04296",
            pdf_url="https://arxiv.org/pdf/2512.04296",
            full_text="Test Full Text",
            tldr="Test TLDR",
            affiliations=["Test Affiliation", "Test Affiliation 2"],
            framework_figure=PNG_BYTES,
            framework_figure_cid="framework-figure-0",
            score=0.5,
        )
    ]


def test_render_email(papers: list[Paper]):
    email_content = render_email(papers)
    assert email_content is not None
    assert "cid:framework-figure-0" in email_content
    assert "Framework Figure" in email_content


def test_build_email_message(config, papers: list[Paper]):
    message = build_email_message(
        config,
        render_email(papers),
        inline_images=[("framework-figure-0", PNG_BYTES)],
    )

    assert message.get_content_type() == "multipart/related"
    assert "framework-figure-0" in message.as_string()


def test_send_email(config, papers: list[Paper], monkeypatch: pytest.MonkeyPatch):
    sent_messages = []

    class FakeSMTP:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def starttls(self):
            return None

        def login(self, sender, password):
            self.sender = sender
            self.password = password

        def sendmail(self, sender, receivers, message):
            sent_messages.append((sender, receivers, message))

        def quit(self):
            return None

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("smtplib.SMTP_SSL", FakeSMTP)

    send_email(
        config,
        render_email(papers),
        inline_images=[("framework-figure-0", PNG_BYTES)],
    )

    assert len(sent_messages) == 1
    assert "Content-ID: <framework-figure-0>" in sent_messages[0][2]
