// Adds an "Admin" button next to "Readme" in the chat page header, for admins only.
// Loaded through custom_js in .chainlit/config.toml. The server decides who is an admin
// (GET /admin/me); the button is only a shortcut, /admin checks again itself.
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
