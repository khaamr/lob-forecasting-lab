export function initializeFilm() {
  const film = document.getElementById('research-film');
  const toggle = document.getElementById('film-toggle');
  const animation = 'assets/research-film.gif';
  const poster = 'assets/research-poster.jpg';
  let paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function update() {
    film.src = paused ? poster : animation;
    toggle.textContent = paused ? '▷' : 'Ⅱ';
    toggle.setAttribute('aria-label', paused ? 'play animation' : 'pause animation');
    toggle.setAttribute('aria-pressed', String(paused));
  }

  toggle.addEventListener('click', () => {
    paused = !paused;
    update();
  });
  update();
}

