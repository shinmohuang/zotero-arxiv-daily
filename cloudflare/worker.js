/**
 * zotero-add —— Cloudflare Worker（点击邮件按钮 → 一键把论文加入 Zotero 待读收藏夹）
 *
 * 邮件里的「+ Zotero 待读」按钮是一个无认证 GET 链接，没法直接带着 Zotero API key
 * 去发认证的 POST。这个 Worker 充当“点击桥接”：它持有 API key（存为 Worker secret），
 * 接到点击后按 arXiv id 抓全元数据，再调 Zotero Web API 把条目（含标题/作者/摘要）
 * 创建到指定收藏夹。
 *
 * 需要配置的 Worker 变量（Settings → Variables and Secrets）：
 *   ZOTERO_ID         —— 你的 Zotero 数字 user id（Settings → Feeds/API 页面可见）
 *   ZOTERO_KEY        —— 带【写权限】的 Zotero API key（注意不是只读的那个）
 *   ADD_TOKEN         —— 自己起的一个口令，按钮链接里带 &token=，防止链接被别人乱用
 *   ZOTERO_COLLECTION —— （可选）默认待读收藏夹 key，如 QJT6TCLR；链接里的 collection 覆盖它
 *
 * 按钮链接形如：
 *   https://<worker>.workers.dev/add?arxiv=2512.04296&token=<ADD_TOKEN>&collection=QJT6TCLR
 */

const ARXIV_API = "https://export.arxiv.org/api/query";
const ZOTERO_API = "https://api.zotero.org";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/add") {
      return page("Not Found", "未知路径，请使用 /add。", false, 404);
    }

    const token = url.searchParams.get("token") || "";
    if (!env.ADD_TOKEN || token !== env.ADD_TOKEN) {
      return page("无效口令", "token 缺失或不匹配，已拒绝。", false, 403);
    }
    if (!env.ZOTERO_ID || !env.ZOTERO_KEY) {
      return page("未配置", "Worker 缺少 ZOTERO_ID / ZOTERO_KEY 变量。", false, 500);
    }

    const arxivId = (url.searchParams.get("arxiv") || "").trim();
    if (!arxivId || !/^[\w.\-/]+$/.test(arxivId)) {
      return page("参数错误", "缺少合法的 arxiv 参数。", false, 400);
    }
    const collection =
      url.searchParams.get("collection") || env.ZOTERO_COLLECTION || "";

    try {
      const meta = await fetchArxivMeta(arxivId);
      const item = buildZoteroItem(meta, arxivId, collection);
      await createZoteroItem(env, item);
      return page(
        "✅ 已加入 Zotero 待读",
        `${escapeHtml(meta.title)}`,
        true,
      );
    } catch (err) {
      return page("❌ 加入失败", escapeHtml(String(err && err.message || err)), false, 502);
    }
  },
};

async function fetchArxivMeta(arxivId) {
  const res = await fetch(`${ARXIV_API}?id_list=${encodeURIComponent(arxivId)}&max_results=1`);
  if (!res.ok) {
    throw new Error(`arXiv 接口返回 ${res.status}`);
  }
  const xml = await res.text();
  const entry = matchOne(xml, /<entry>([\s\S]*?)<\/entry>/i);
  if (!entry) {
    throw new Error("arXiv 未返回该论文，请检查 id");
  }

  const title = clean(matchOne(entry, /<title>([\s\S]*?)<\/title>/i));
  const summary = clean(matchOne(entry, /<summary>([\s\S]*?)<\/summary>/i));
  const published = clean(matchOne(entry, /<published>([\s\S]*?)<\/published>/i));
  const doi = clean(matchOne(entry, /<arxiv:doi>([\s\S]*?)<\/arxiv:doi>/i));

  const authors = [];
  const nameRe = /<author>\s*<name>([\s\S]*?)<\/name>/gi;
  let m;
  while ((m = nameRe.exec(entry)) !== null) {
    const name = clean(m[1]);
    if (name) authors.push(name);
  }

  if (!title) {
    throw new Error("解析 arXiv 元数据失败（无标题）");
  }
  return {
    title,
    summary,
    authors,
    doi,
    date: (published || "").slice(0, 10), // YYYY-MM-DD
  };
}

function buildZoteroItem(meta, arxivId, collection) {
  return {
    itemType: "preprint",
    title: meta.title,
    creators: meta.authors.map(splitName),
    abstractNote: meta.summary,
    repository: "arXiv",
    archiveID: `arXiv:${arxivId}`,
    DOI: meta.doi || "",
    url: `https://arxiv.org/abs/${arxivId}`,
    date: meta.date,
    libraryCatalog: "arXiv.org",
    collections: collection ? [collection] : [],
    tags: [{ tag: "to-read" }],
  };
}

function splitName(fullName) {
  const parts = fullName.trim().split(/\s+/);
  if (parts.length === 1) {
    return { creatorType: "author", firstName: "", lastName: parts[0] };
  }
  return {
    creatorType: "author",
    lastName: parts[parts.length - 1],
    firstName: parts.slice(0, -1).join(" "),
  };
}

async function createZoteroItem(env, item) {
  const res = await fetch(`${ZOTERO_API}/users/${env.ZOTERO_ID}/items`, {
    method: "POST",
    headers: {
      "Zotero-API-Key": env.ZOTERO_KEY,
      "Zotero-API-Version": "3",
      "Content-Type": "application/json",
    },
    body: JSON.stringify([item]),
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`Zotero 接口 ${res.status}: ${text.slice(0, 200)}`);
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("Zotero 返回非 JSON");
  }
  if (data.failed && Object.keys(data.failed).length > 0) {
    const first = data.failed[Object.keys(data.failed)[0]];
    throw new Error(`Zotero 拒绝：${first && first.message || "未知原因"}`);
  }
  return data;
}

// ---- 小工具 ----
function matchOne(text, re) {
  const m = text.match(re);
  return m ? m[1] : "";
}

function clean(s) {
  return unescapeXml(String(s || ""))
    .replace(/\s+/g, " ")
    .trim();
}

function unescapeXml(s) {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d) => String.fromCodePoint(parseInt(d, 10)))
    .replace(/&amp;/g, "&");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function page(title, body, ok, status = 200) {
  const color = ok ? "#2e7d32" : "#c62828";
  const html = `<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title></head>
<body style="font-family:-apple-system,Arial,sans-serif;background:#f5f5f5;margin:0;padding:48px 16px;">
<div style="max-width:520px;margin:0 auto;background:#fff;border-radius:12px;padding:28px;box-shadow:0 2px 12px rgba(0,0,0,.08);">
<h1 style="font-size:20px;color:${color};margin:0 0 12px;">${escapeHtml(title)}</h1>
<p style="font-size:15px;color:#444;line-height:1.6;margin:0;">${body}</p>
<p style="font-size:12px;color:#999;margin-top:20px;">可以关闭此页。</p>
</div></body></html>`;
  return new Response(html, {
    status,
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
}
