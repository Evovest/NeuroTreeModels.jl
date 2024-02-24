import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
// import footnote from "markdown-it-footnote";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_WITH_DOCUMENTER_VITEPRESS_BASE_URL_WITH_TRAILING_SLASH',
  title: "NeuroTreeModels",
  description: "Docs for NeuroTreeModels.jl",
  lastUpdated: true,
  cleanUrls: true,
  ignoreDeadLinks: true,

  head: [['link', { rel: 'icon', href: '/base/favicon.ico' }]],

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
        md.use(mathjax3)
      // md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    }
  },
  themeConfig: {
    outline: 'deep',
    // https://vitepress.dev/reference/default-theme-config
    logo: { src: '/evo-chevron.png', width: 24, height: 24 },
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Quick start', link: '/quick-start' },
      { text: 'Design', link: '/design' },
      { text: 'Models', link: '/models' },
      { text: 'API', link: '/API' },
      {
        text: 'Tutorials',
        items: [
          { text: 'Regression', link: '/tutorials/regression-boston' },
          { text: 'Logistic', link: '/tutorials/logistic-titanic' },
          { text: 'Classification', link: '/tutorials/classification-iris' },
        ]
      },
    ],
    sidebar: [
      { text: 'Quick start', link: '/quick-start' },
      { text: 'Design', link: '/design' },
      { text: 'Models', link: '/models' },
      { text: 'API', link: '/API' },
      {
        text: 'Tutorials',
        items: [
          { text: 'Regression', link: '/tutorials/regression-boston' },
          { text: 'Logistic', link: '/tutorials/logistic-titanic' },
          { text: 'Classification', link: '/tutorials/classification-iris' },
        ]
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Evovest/NeuroTreeModels.jl' }
    ]
  }
})