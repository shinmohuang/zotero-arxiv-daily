# 一键加入 Zotero 待读（Cloudflare Worker）

邮件里每篇论文的「**+ Zotero 待读**」按钮点击后，会打到这个 Worker：它按 arXiv id
抓取论文元数据（标题/作者/摘要/日期），再调 Zotero Web API 把条目创建到你的待读收藏夹。

之所以要这个中转：邮件按钮只是一个无认证的 GET 链接，没法直接带着 Zotero API key
去发认证的 POST，key 也不能明文放进邮件。Worker 持有 key（存为 secret），安全地完成这一步。

> 免费版 Workers 每天 10 万次请求，远超日常点击量，**不花钱、不用绑卡、不用买域名**。

---

## 一、准备 Zotero 凭据

1. **user id**：打开 <https://www.zotero.org/settings/security#applications>（旧版在 Feeds/API 页），
   页面上的纯数字「Your userID」就是 `ZOTERO_ID`。
2. **API key（必须带写权限）**：同页 New Private Key →
   勾选 **Allow library access** 和 **Allow write access** → 创建后复制，作为 `ZOTERO_KEY`。
   ⚠️ 仓库里原来发邮件用的那个 key 是**只读**的，这里不能复用，要新建一个带写权限的。
3. **待读收藏夹 key**：在 Zotero 里那个收藏夹的链接 `zotero://select/library/collections/QJT6TCLR`
   末尾的 `QJT6TCLR` 就是 `ZOTERO_COLLECTION`。

## 二、部署 Worker

两种任选其一。注意：**不要走 “Pages / Upload 上传文件” 那条路**——它不支持直接传 `.js`，
会报 “At least one JavaScript file was found. Please use 'wrangler deploy' instead”。

### 方式 A：网页后台粘代码（免命令行）

1. 登录 <https://dash.cloudflare.com> → **Workers & Pages** → **Create** → 选 **Workers** 标签页 → **Create Worker**。
2. 起名 `zotero-add` → Deploy（先部署默认 Hello World 模板）。
3. 进入这个 Worker → **Edit code** → 把 [`worker.js`](./worker.js) 整段粘贴覆盖 → **Deploy**。
4. **Settings → Variables and Secrets** 添加：

   | 名称 | 类型 | 值 |
   |---|---|---|
   | `ZOTERO_ID` | Secret | 你的数字 user id |
   | `ZOTERO_KEY` | Secret | 带写权限的 API key |
   | `ADD_TOKEN` | Secret | 自己起一个口令，如 `a8f3k2` |
   | `ZOTERO_COLLECTION` | Text | `QJT6TCLR` |

   保存后会自动重新部署。

### 方式 B：wrangler 命令行（更稳，报错提示的就是这条）

本目录已带 [`wrangler.toml`](./wrangler.toml)。在 `cloudflare/` 目录下：

```bash
npx wrangler login                  # 浏览器授权一次
npx wrangler deploy                 # 部署，结束打印 Worker 地址
npx wrangler secret put ZOTERO_ID   # 粘贴数字 user id
npx wrangler secret put ZOTERO_KEY  # 粘贴带写权限的 API key
npx wrangler secret put ADD_TOKEN   # 自己起一个口令
```

`ZOTERO_COLLECTION` 已在 `wrangler.toml` 的 `[vars]` 里，改 collection 改文件后重新 `npx wrangler deploy` 即可。

部署完记下 Worker 地址，形如 `https://zotero-add.<你的子域>.workers.dev`。

## 三、自测

浏览器打开（把 id / token 换成你的）：

```
https://zotero-add.<你的子域>.workers.dev/add?arxiv=2512.04296&token=a8f3k2
```

看到「✅ 已加入 Zotero 待读」，并在 Zotero 待读收藏夹里出现该论文即成功。

## 四、接到邮件按钮

在 `config/custom.yaml`（GitHub Actions 里是 `CUSTOM_CONFIG` 变量）里加上：

```yaml
zotero_add:
  enabled: true
  endpoint: https://zotero-add.<你的子域>.workers.dev/add
  token: a8f3k2          # 与 Worker 的 ADD_TOKEN 一致
  collection: QJT6TCLR   # 留空则用 Worker 默认
```

之后每封邮件里每篇 arXiv 论文的 PDF 按钮旁就会出现「+ Zotero 待读」，点一下即入库。

---

### 备注

- 按钮链接里会带上 `token`。邮件只发给你自己，风险很低；`ADD_TOKEN` 主要用来挡住
  对 `*.workers.dev` 的随机扫描。想更稳可定期换 token（Worker secret 与 config 同步改即可）。
- 仅对能解析出 arXiv id 的论文显示按钮（biorxiv/medrxiv 等不显示）。
- 条目类型用 Zotero 的 `preprint`，并打上 `to-read` 标签。
