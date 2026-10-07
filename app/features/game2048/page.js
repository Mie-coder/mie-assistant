// Read visible facts from the loaded store; moves go through the game's normal
// key handler. Never edit tiles, scores, RNG, storage, or call store.move directly.
async (action = null) => {
  if (location.origin !== "https://play2048.co" || location.pathname !== "/") {
    throw new Error("Unexpected game page");
  }
  // Reduce the decorative pulse only in this owned game tab. Preserve the
  // input blocker, control status, stop button, and browser debugger indicator.
  const overlay = document.querySelector("browser-skill-overlay")?.shadowRoot;
  if (overlay && !overlay.querySelector("#mie-2048-reduced-motion")) {
    const style = document.createElement("style");
    style.id = "mie-2048-reduced-motion";
    style.textContent = '[data-slot="control-overlay"] { animation: none !important; '
      + 'box-shadow: inset 0 0 20px 4px rgba(249,115,22,.16) !important; }';
    overlay.append(style);
  }
  const script = [...document.scripts].map(s => s.src).find(src => {
    const url = new URL(src, location.href);
    return url.origin === location.origin && /^\/assets\/index-[\w-]+\.js$/.test(url.pathname);
  });
  if (!script) throw new Error("Game module not found");
  const module = await import(script);
  const stores = Object.values(module).filter(value => value && typeof value === "object"
    && typeof value.subscribe === "function" && typeof value.move === "function"
    && typeof value.resetWithConfirm === "function");
  if (stores.length !== 1) throw new Error("Game interface changed");
  const read = () => {
    let result;
    const unsubscribe = stores[0].subscribe(state => {
      // Never read the internal random seed or history.
      result = {id: state.id, state: state.state, score: state.score,
        moveCount: state.moveCount,
        board: state.board.map(row => row.map(tile => tile === null ? 0 : tile.value))};
    });
    unsubscribe();
    return result;
  };
  const current = read();
  if (!action) return current;
  const expected = action.expected;
  if (!expected || ["id", "state", "score", "moveCount"].some(k => current[k] !== expected[k])
      || JSON.stringify(current.board) !== JSON.stringify(expected.board)) {
    return {error: "board_changed"};
  }
  if (action.restart === true) {
    if (current.state !== "gameOver") return {error: "restart_requires_game_over"};
    // The already-observed terminal button. BrowserSkill's input protection
    // covers physical clicks, so dispatch the DOM button's normal click event.
    // Keep the protection in place; never call the store's reset methods.
    const buttons = [...document.querySelectorAll("button")].filter(b =>
      /^(Play Again|New Game)$/.test(b.textContent.trim()) && !b.disabled
      && b.getBoundingClientRect().width > 0 && b.getBoundingClientRect().height > 0);
    if (buttons.length !== 1) return {error: "restart_button_unavailable"};
    buttons[0].click();
    // Only a new-game transition waits for mounting; no retry of the click.
    for (let attempt = 0; attempt < 20 && read().id === current.id; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    return read();
  }
  const keys = {left: "ArrowLeft", right: "ArrowRight", up: "ArrowUp", down: "ArrowDown"};
  const key = keys[action.direction];
  if (!key) return {error: "invalid_choice"};
  if (["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
    return {error: "input_focused"};
  }
  // No await between the guard and input. Runtime.evaluate avoids moving browser
  // focus on each turn, and the game updates its store synchronously on keydown.
  for (const type of ["keydown", "keyup"]) {
    window.dispatchEvent(new KeyboardEvent(type, {key, code: key, bubbles: true, cancelable: true}));
  }
  return read();
}
