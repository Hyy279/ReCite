(function () {
  "use strict";

  /* ---------- scroll progress ---------- */
  var progress = document.getElementById("scroll-progress");

  function updateProgress() {
    if (!progress) return;
    var doc = document.documentElement;
    var max = doc.scrollHeight - window.innerHeight;
    var ratio = max > 0 ? Math.min(1, window.scrollY / max) : 0;
    progress.style.width = (ratio * 100).toFixed(2) + "%";
  }

  /* ---------- nav active state ---------- */
  var navLinks = Array.prototype.slice.call(
    document.querySelectorAll(".topnav-links a")
  );
  var sections = navLinks
    .map(function (link) {
      var id = link.getAttribute("href");
      return id && id.charAt(0) === "#"
        ? document.getElementById(id.slice(1))
        : null;
    })
    .filter(Boolean);

  function updateNav() {
    var y = window.scrollY + 120;
    var active = null;
    for (var i = 0; i < sections.length; i += 1) {
      if (sections[i].offsetTop <= y) active = sections[i].id;
    }
    navLinks.forEach(function (link) {
      var target = link.getAttribute("href");
      link.classList.toggle(
        "active",
        target === "#" + active || (!target || target === "#top")
      );
    });
  }

  function onScroll() {
    updateProgress();
    updateNav();
    updateToTop();
  }

  /* ---------- back to top ---------- */
  var toTop = document.getElementById("to-top");

  function updateToTop() {
    if (!toTop) return;
    var visible = window.scrollY > 720;
    toTop.classList.toggle("visible", visible);
  }

  if (toTop) {
    toTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
      toTop.focus({ preventScroll: true });
    });
  }

  var ticking = false;
  window.addEventListener("scroll", function () {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        onScroll();
        ticking = false;
      });
      ticking = true;
    }
  });

  window.addEventListener("resize", function () {
    updateProgress();
  });

  onScroll();

  /* ---------- reveal on scroll ---------- */
  var revealTargets = Array.prototype.slice.call(
    document.querySelectorAll(
      ".motivation-copy, .motivation-figure, .abstract-layout, " +
      ".module-grid, .stat-grid, .figure-slot, .insight-media, " +
      ".takeaway-note, .tables-duo, " +
      ".figure-row, .table-block"
    )
  );

  function revealVisible(entries, observer) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }

  if ("IntersectionObserver" in window && revealTargets.length) {
    var revealObserver = new IntersectionObserver(revealVisible, {
      threshold: 0.1,
      rootMargin: "0px 0px -40px 0px"
    });

    revealTargets.forEach(function (el) {
      el.classList.add("reveal");
      revealObserver.observe(el);
    });
  }

  /* ---------- mobile menu ---------- */
  var menuButton = document.getElementById("menu-toggle");
  var navList = document.getElementById("nav-links");

  if (menuButton && navList) {
    menuButton.addEventListener("click", function () {
      var open = navList.classList.toggle("open");
      menuButton.setAttribute("aria-expanded", open ? "true" : "false");
    });

    navLinks.forEach(function (link) {
      link.addEventListener("click", function () {
        navList.classList.remove("open");
        menuButton.setAttribute("aria-expanded", "false");
      });
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        navList.classList.remove("open");
        menuButton.setAttribute("aria-expanded", "false");
        menuButton.focus();
      }
    });
  }

  /* ---------- BibTeX copy ---------- */
  var copyBtn = document.getElementById("bib-copy-btn");
  var content = document.getElementById("bib-content");

  if (copyBtn && content) {
    copyBtn.addEventListener("click", function () {
      var text = (content.textContent || "").replace(/^\s+|\s+$/g, "");
      var original = copyBtn.innerHTML;

      function copied() {
        copyBtn.textContent = "Copied";
        copyBtn.style.color = "#059669";
        copyBtn.style.borderColor = "#059669";
        window.setTimeout(function () {
          copyBtn.innerHTML = original;
          copyBtn.style.color = "";
          copyBtn.style.borderColor = "";
        }, 1800);
      }

      function fallbackCopy() {
        var textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand("copy");
          copied();
        } catch (e) {
          /* noop */
        }
        document.body.removeChild(textarea);
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(copied, fallbackCopy);
      } else {
        fallbackCopy();
      }
    });
  }
})();
