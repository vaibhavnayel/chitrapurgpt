// Small additions to the chat page, loaded through custom_js in .chainlit/config.toml.

// 1. An "Admin" button next to "Readme" in the header, for admins only. The server decides who is
// an admin (GET /admin/me); the button is only a shortcut, /admin checks again itself.
(function () {
  var BUTTON_ID = "admin-button";
  var CLASSES = "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium " +
    "ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground h-10 px-4 py-2";
  var isAdmin = false;

  function addButton() {
    if (!isAdmin || document.getElementById(BUTTON_ID)) return;
    var anchor = document.getElementById("readme-button") || document.getElementById("theme-toggle");
    if (!anchor || !anchor.parentNode) return;
    var link = document.createElement("a");
    link.id = BUTTON_ID;
    link.href = "/admin";
    link.className = CLASSES;
    link.title = "Admin dashboard: manage who can sign in and read chats";
    link.textContent = "Admin";
    anchor.parentNode.insertBefore(link, anchor);
  }

  function check() {
    fetch("/admin/me", { credentials: "same-origin", cache: "no-store" })
      .then(function (r) { return r.ok ? r.json() : { admin: false }; })
      .then(function (d) {
        isAdmin = !!(d && d.admin);
        if (isAdmin) addButton();
        else {
          var old = document.getElementById(BUTTON_ID);
          if (old) old.remove();
        }
      })
      .catch(function () {});
  }

  // The page is a React app that re-draws its header, so put the button back whenever it's gone.
  new MutationObserver(addButton).observe(document.documentElement, { childList: true, subtree: true });
  check();
  // Re-check after signing in or out without a full page reload.
  var lastHeader = null;
  setInterval(function () {
    var header = document.getElementById("header");
    if (header !== lastHeader) { lastHeader = header; check(); }
  }, 2000);
})();

// 2. The chat settings button (a bare gear icon in the message box) only holds the collections
// picker (kb_picker.py), and only for people with more than one collection, so give it a label.
(function () {
  var LABEL = "Collections";
  function label() {
    var button = document.getElementById("chat-settings-open-modal");
    if (!button || button.dataset.cgptLabelled) return;
    button.dataset.cgptLabelled = "1";
    button.title = "Choose which collections to search";
    button.setAttribute("aria-label", "Choose which collections to search");
    button.style.width = "auto";
    button.style.padding = "0 10px 0 6px";
    button.style.gap = "6px";
    var text = document.createElement("span");
    text.textContent = LABEL;
    text.style.fontSize = "13px";
    text.style.fontWeight = "500";
    button.appendChild(text);
  }
  // The message box is re-drawn by React, so label the button whenever it (re)appears.
  new MutationObserver(label).observe(document.documentElement, { childList: true, subtree: true });
  label();
})();

// 3. In the collections picker, keep "All my collections" and the individual boxes in step while the panel is
// open: ticking or unticking "All" does the same to every collection, and "All" is ticked exactly when every
// collection is. (kb_picker.update tidies up the same way when the panel is confirmed.)
(function () {
  var syncing = false;
  function isOn(box) { return box.getAttribute("data-state") === "checked"; }
  function collections() {
    return Array.prototype.slice.call(document.querySelectorAll('#chat-settings [role="checkbox"][id^="kb:"]'));
  }
  document.addEventListener("click", function (e) {
    if (syncing || !e.target.closest) return;
    var box = e.target.closest('#chat-settings [role="checkbox"]');
    if (!box || !(box.id === "kb_all" || box.id.indexOf("kb:") === 0)) return;
    setTimeout(function () {             // after the clicked box has changed
      var all = document.getElementById("kb_all");
      if (!all) return;
      var items = collections();
      syncing = true;
      try {
        if (box === all) {
          items.forEach(function (b) { if (isOn(b) !== isOn(all)) b.click(); });
        } else if (isOn(all) !== items.every(isOn)) {
          all.click();
        }
      } finally { syncing = false; }
    }, 0);
  }, true);
})();
