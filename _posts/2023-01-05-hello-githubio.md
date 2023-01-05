---
layout: post
title: "Hello World, to github.io"
subtitle: 'Why and how I set up github.io as my homepage.'
author: "CaveSpider"
header-style: text
tags:
  - 技术
  - 杂谈
  - 开发环境
---

很早就有搭个博客主页放简历放日常放一些杂七杂八的东西的想法，鸽了两年直到今天才步入了正轨。这篇是本站的第一篇文章，加上发布日期是 2023 年初，颇有些纪念意味。一件事刚开始做的时候总是需要些雄心壮志的，我因此粗略地想了一下这个网站可以写点什么。学术？Publication 肯定是重要的一部分，奈何本人现在还没有什么值得写的东西，这部分就留到后面再补。技术分享？我是学计算机的，技术肯定是本站的核心话题，但回顾我的本科三年，从刚入门计算机到现在，也并没有在某一方面钻研出什么特长或门道。想想学过的东西，虽然很多但也很杂，一时想不起什么可以总结的脉络，这部分就暂且随想随写，平时碰到了什么问题，有意思的话我就把解决方案放上来，说不定写的多了会找到些门道。日常？我不想把本站变得太过严肃，平时碰到什么好玩的事或者有什么好玩的想法也尽量放上来，但我会尽量把专业和非专业的东西分开，比如之后给网站多分几个标签之类的。

总之，在我的想象中，本站应该是一个以学术经历和技术分享为主，生活日常和闲谈随记为辅的小站。从今天开始我也会比较经常地维护它，尽量做到周更或者月更，希望我能坚持下来。

第一篇文章算是个试水，就简单记录一下我为什么选择 `github.io` 作为个人主页的平台，以及在搭建过程中遇到了哪些问题。

## Why `github.io`?

`github.io` 即 [GitHub Pages](https://pages.github.com)，是全球最大~~同性交友网站~~代码托管平台 [GitHub](https://github.com) 自 2015 年起推出的一项服务。你只需要在自己的账号下创建一个名为 `<username>.github.io` 的仓库，GitHub Pages 就会自动为这个仓库部署一个博客网站，默认的域名就是 `<username>.github.io`，常用 GitHub 的话极为方便。这个网站由 [Jekyll](https://jekyllrb.com) 驱动部署。它本身是个静态网站，没有后端数据库，因此发布的内容仅由仓库本身决定，足够方便也可以满足日常记录博文的需求。它本身足够轻量，唯一的限制条件是仓库大小不能超过 1G，做不了过于复杂的事情，但对于记录些文字还是绰绰有余了。因为仓库的内容全部由开发者决定，因此它的定制性很强，你大可以手撸 `html` + `css` 不依赖任何轮子从头写一个网站来锻炼技术。但对于像我这样的前端小白，Jekyll 背后庞大的社区提供了很多成熟的主题、模板和开发支持，可以从开源项目中直接魔改出一个好看的个人网站。本站就是来自 Huxpro 大神的开源项目 [Hux Blog](https://github.com/huxpro/huxpro.github.io)。

总结一下原因就是，常用 Github、静态网站足够满足需求、社区庞大方便魔改、免费，因此选择了 `github.io`，很香。

## How to set up your `github.io`?

使用 Jekyll 开源项目来魔改的话，几乎不需要写什么代码。我本地的开发环境是 MacOS Monterey + M1 Pro Apple Silicon，下面就以魔改 [Hux Blog](https://github.com/huxpro/huxpro.github.io) 为例简单梳理一下搭建步骤：

* 首先准备好浏览器和终端均能科学上网的网络环境。
    * 浏览器可以使用 shawdowsocks 挂 VPN。
    * 配置好 shadowsocks 后可以参考 [mac 终端实现翻墙](https://kerminate.me/2018/10/22/mac-终端实现翻墙/)实现终端科学上网。
* 创建 `github.io` 仓库：根据 [GitHub Pages with Jekyll](https://docs.github.com/en/pages) 的前两步创建自己的 `github.io` 仓库。
    * 设置好域名和 public source 之后在空仓库里创建一个简单的首页，比如 `$ echo "Hello World" > index.html`，访问 `<username>.github.io` 域名，能看到文件内容就算成功。
* 安装最新的 Ruby，不要使用 mac 自带的 Ruby，因为版本太低。
    * 参考 [Ruby 安装教程](https://www.moncefbelyamani.com/how-to-install-xcode-homebrew-git-rvm-ruby-on-mac/)，使用 chruby 管理 Ruby 的不同版本，走完全部流程大约需要一个小时。
    * 不需要使用其中付费的 [Ruby on Mac](https://www.rubyonmac.dev/?utm_campaign=install-ruby-guide)，脚本，除非你想花钱。
    * 流程中绝大多数步骤跟着走就行，除了有一步 `ruby-install 3.1.3` 大概率会出错，这是因为它用到了 `wget`，而 `wget` 因为域名污染无法访问 `raw.githubusercontent.com`，此时需要用 [IP 查询工具](https://www.ipaddress.com) 查到 `raw.githubusercontent.com` 的真实 IP 地址，例如 `xxx.xxx.xxx.xxx`，然后把 `xxx.xxx.xxx.xxx raw.githubusercontent.com ` 这一行加到本地 `/etc/hosts` 文件的最后一行，然后再 `ruby-install 3.1.3` 就可以成功。
* 使用 Ruby 安装 Jekyll，参考 [Jekyll Installation](https://jekyllrb.com/docs/installation/)。
* Clone [Hux Blog](https://github.com/huxpro/huxpro.github.io) 到自己的仓库，按照 README 中所述此时项目可直接运行，但内容还是 Hux 自己的内容，需要参考 `_doc/Manual.md` 中的内容魔改为自己的网站内容，包括但不限于：
    * 改一些静态的图片背景、网站链接之类的资源。
    * 使用 [Disqus](https://disqus.com) 在网站中添加第三方评论功能。
    * 使用 [Google Analytics](https://analytics.google.com/analytics/web/#/) 追踪网站流量。
    * 重写 `_includes/about/` 下的中英文个人简介。
    * 删除掉 `_posts/` 中原有的文章，替换成你自己的。
    * 其他魔改自己网站的需求。

经过以上步骤，一个初具规模的个人网站雏形就有了，也就是我目前达到的状态。之后的事情就是根据自己的需求魔改网站功能，并定期发博文了。搭建的过程本身还是挺有意思的，而且不需要任何计算机知识和前端技术。之后要是有机会我一定好好学前端把它魔改得更像样一点（又一个 flag）。
