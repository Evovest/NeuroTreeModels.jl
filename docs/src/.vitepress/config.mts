import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
// import footnote from "markdown-it-footnote";
// https://vitepress.dev/reference/site-config

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
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
    }
  ]
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker',
  }
]

export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: "Differentiable tree-based models for tabular data",
  lastUpdated: true,
  cleanUrls: true,
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', { src: `${getBaseRepository(baseTemp.base)}versions.js` }],
    ['script', { src: `${baseTemp.base}siteinfo.js` }]
  ],
  vite: {
    build: {
      assetsInlineLimit: 0, // so we can tell whether we have created inlined images or not, we don't let vite inline them
    },
    optimizeDeps: {
      exclude: [
        'vitepress',
      ],
    }
  },

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
        md.use(mathjax3)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    }
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
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
      }
    ],
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
  }
})
