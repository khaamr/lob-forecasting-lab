"""
Build the website's readme page from the repository guide.
"""

from pathlib import Path
import re
import markdown
from bs4 import BeautifulSoup, Comment, Doctype


ROOT = Path(__file__).resolve().parents[1]


def build_readme():
    """
    Render the readme using the shared website design.
    """
    source = (ROOT / 'README.md').read_text()
    source = source.replace('<div align="center">', '').replace('</div>', '')
    source = re.sub(r'```mermaid\n.*?```', '<p><strong>book updates / clean and align / 100 × 40 windows / purged splits / forecasts / classification and timing</strong></p>', source, flags=re.S)
    content = markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc'])
    content = content.replace('<table>', '<div class="table-scroll"><table>').replace('</table>', '</table></div>')
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="The LOB forecasting reproduction: findings, methods, setup, use cases and provenance."><meta name="theme-color" content="#0b1015"><title>Readme · LOB Forecasting Lab</title><link rel="icon" href="assets/mark.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/site.css"><script type="module" src="assets/readme.js"></script></head><body><a class="skip" href="#main">Skip to content</a>
<header class="header"><div class="shell header-inner"><a class="brand" href="index.html">LOB<span class="brand-light">/</span>lab</a><nav aria-label="Main navigation"><a href="index.html#benchmarks">Benchmarks</a><a href="research_dashboard.html">Dashboard</a><a href="readme.html" aria-current="page">Readme</a></nav><a class="button button-small button-outline" href="https://github.com/khaamr/lob-forecasting-lab">GitHub</a></div></header>
<main class="shell readme-layout" id="main"><aside class="readme-sidebar"><p class="eyebrow">THE RESEARCH GUIDE</p><a href="#explore-the-study">Explore the study</a><a href="#from-book-updates-to-decisions">The method</a><a href="#what-the-saved-experiments-show">Saved results</a><a href="#extensions-beyond-the-benchmark">Extensions</a><a href="#reproduce">Reproduce</a><a href="#data-and-provenance">Data and provenance</a><a href="#practical-use-cases">Use cases</a><a href="#website-and-media">Website and media</a></aside><article class="readme-content"><a class="back-link" href="index.html">BACK TO THE LAB</a>'''
    page += content
    page += '</article></main><footer class="shell footer"><a class="brand" href="index.html">LOB<span class="brand-light">/</span>lab</a><p>Public-data reproduction & extensions<br>Research prototype · no live market feed</p><div><a href="README.md">Markdown source</a><a href="https://doi.org/10.1080/14697688.2025.2522911">Original article</a><a href="research_dashboard.html">Full dashboard</a></div></footer></body></html>'
    soup = BeautifulSoup(page, 'html.parser')
    image = soup.find('img', src='assets/research-film.gif')
    if image is not None:
        image['id'] = 'research-film'
        image['width'] = '800'
        image['height'] = '450'
        toggle = soup.new_tag('button', attrs={
            'id': 'film-toggle',
            'type': 'button',
            'class': 'button button-small button-outline',
            'aria-label': 'pause animation',
        })
        toggle.string = 'Ⅱ'
        image.parent.insert_after(toggle)
    else:
        soup.find('script', src='assets/readme.js').decompose()
    for node in list(soup.find_all(string=True)):
        if isinstance(node, (Comment, Doctype)) or node.find_parent(['script', 'style', 'code', 'pre']):
            continue
        node.replace_with(str(node).lower())
    for tag in soup.find_all(True):
        for attribute in ('aria-label', 'alt', 'title'):
            if tag.has_attr(attribute):
                tag[attribute] = tag[attribute].lower()
    soup.title.string = 'readme · lob forecasting lab'
    soup.find('meta', attrs={'name':'description'})['content'] = 'lob forecasting reproduction: findings, methods, setup, use cases and provenance.'
    (ROOT / 'readme.html').write_text(soup.prettify())


if __name__ == '__main__':
    build_readme()
