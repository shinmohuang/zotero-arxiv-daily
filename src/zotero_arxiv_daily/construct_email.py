from .protocol import Paper
import math
import re
from urllib.parse import urlencode

_ARXIV_ID_PATTERN = re.compile(r"arxiv\.org/abs/([^\s?#]+)", re.IGNORECASE)


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def get_framework_figure_html(framework_figure_cid:str | None) -> str:
    if framework_figure_cid is None:
        return ""
    return f"""
    <tr>
        <td style="padding: 12px 0;">
            <div style="font-size: 12px; font-weight: bold; color: #555; padding-bottom: 8px;">
                Framework Figure
            </div>
            <img src="cid:{framework_figure_cid}" alt="Framework figure extracted from paper PDF" style="display: block; width: 100%; max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;">
        </td>
    </tr>
"""

# 邮件里的链接无法直接带着 Zotero API key 发认证 POST，所以「+ Zotero 待读」按钮
# 指向一个 Cloudflare Worker（见 cloudflare/worker.js）：点击 → Worker 按 arXiv id
# 抓元数据 → 调 Zotero API 把论文（含标题/作者/摘要）创建到待读收藏夹。
# 按钮链接形如 https://<worker>/add?arxiv=<id>&token=<token>&collection=<key>


def extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _ARXIV_ID_PATTERN.search(url)
    return match.group(1) if match else None


def build_zotero_add_url(
    endpoint: str | None,
    arxiv_id: str | None,
    token: str | None = None,
    collection: str | None = None,
) -> str | None:
    if not endpoint or not arxiv_id:
        return None
    params = {"arxiv": arxiv_id}
    if token:
        params["token"] = token
    if collection:
        params["collection"] = collection
    separator = "&" if "?" in endpoint else "?"
    return f"{endpoint}{separator}{urlencode(params)}"


def get_zotero_button_html(add_url: str | None) -> str:
    if not add_url:
        return ""
    return (
        f'<a href="{add_url}" target="_blank" '
        'style="display: inline-block; text-decoration: none; font-size: 14px; '
        'font-weight: bold; color: #fff; background-color: #5bc0de; '
        'padding: 8px 16px; border-radius: 4px; margin-left: 8px;" '
        'title="一键把论文加入 Zotero 待读收藏夹">'
        '+ Zotero 待读</a>'
    )


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, framework_figure_cid:str | None=None, add_url:str | None=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>
    {framework_figure}

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>{zotero_button}
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title,
        authors=authors,
        rate=rate,
        tldr=tldr,
        pdf_url=pdf_url,
        affiliations=affiliations,
        framework_figure=get_framework_figure_html(framework_figure_cid),
        zotero_button=get_zotero_button_html(add_url),
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(
    papers: list[Paper],
    add_endpoint: str | None = None,
    add_token: str | None = None,
    add_collection: str | None = None,
) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        add_url = build_zotero_add_url(
            add_endpoint,
            extract_arxiv_id(p.url),
            token=add_token,
            collection=add_collection,
        )
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.pdf_url or p.url,
                affiliations,
                p.framework_figure_cid,
                add_url=add_url,
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
