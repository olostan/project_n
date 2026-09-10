/* ==============================================================================
   Project N: Interactive Diagram Zoom & Lightbox
   Enhances readability for complex clinical workflows and architecture diagrams
   ============================================================================== */

(function() {
  let modal = null;
  let modalBody = null;
  let currentScale = 1.0;

  function createModal() {
    if (document.getElementById("diagram-modal")) return;

    modal = document.createElement("div");
    modal.id = "diagram-modal";
    modal.className = "diagram-modal";
    modal.innerHTML = `
      <div class="diagram-modal-header">
        <div class="diagram-modal-title">🔍 Project N Architecture & Clinical Workflow Inspector</div>
        <div class="diagram-modal-controls">
          <button class="diagram-modal-btn" id="modal-zoom-out">🔍 Zoom Out (-)</button>
          <button class="diagram-modal-btn" id="modal-zoom-reset">100% Reset</button>
          <button class="diagram-modal-btn" id="modal-zoom-in">🔍 Zoom In (+)</button>
          <button class="diagram-modal-close" id="modal-close">✕ Close</button>
        </div>
      </div>
      <div class="diagram-modal-body" id="diagram-modal-body"></div>
    `;

    document.body.appendChild(modal);
    modalBody = document.getElementById("diagram-modal-body");

    document.getElementById("modal-close").addEventListener("click", closeModal);
    document.getElementById("modal-zoom-in").addEventListener("click", () => adjustZoom(0.2));
    document.getElementById("modal-zoom-out").addEventListener("click", () => adjustZoom(-0.2));
    document.getElementById("modal-zoom-reset").addEventListener("click", () => {
      currentScale = 1.0;
      applyZoom();
    });

    modal.addEventListener("click", (e) => {
      if (e.target === modal || e.target === modalBody) {
        closeModal();
      }
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("active")) {
        closeModal();
      }
    });
  }

  function adjustZoom(delta) {
    currentScale = Math.max(0.4, Math.min(3.0, currentScale + delta));
    applyZoom();
  }

  function applyZoom() {
    const svg = modalBody ? modalBody.querySelector("svg") : null;
    if (svg) {
      svg.style.transform = `scale(${currentScale})`;
    }
  }

  function openModal(svgElement) {
    createModal();
    modalBody.innerHTML = "";
    const clone = svgElement.cloneNode(true);
    clone.style.width = "auto";
    clone.style.height = "auto";
    clone.style.maxWidth = "none";
    modalBody.appendChild(clone);
    currentScale = 1.0;
    applyZoom();
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    if (modal) {
      modal.classList.remove("active");
      document.body.style.overflow = "";
    }
  }

  function attachDiagramControls() {
    createModal();
    const diagrams = document.querySelectorAll(".mermaid");
    diagrams.forEach((diagram) => {
      if (diagram.dataset.hasControls) return;

      const svg = diagram.querySelector("svg");
      if (!svg) return;

      diagram.dataset.hasControls = "true";

      // Create toolbar
      const toolbar = document.createElement("div");
      toolbar.className = "diagram-toolbar";
      
      const expandBtn = document.createElement("button");
      expandBtn.type = "button";
      expandBtn.className = "diagram-expand-btn";
      expandBtn.innerHTML = "🔍 Click to Expand / Zoom Diagram";
      expandBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openModal(svg);
      });

      toolbar.appendChild(expandBtn);
      diagram.parentNode.insertBefore(toolbar, diagram);

      // Also open when double clicking diagram
      diagram.addEventListener("dblclick", () => openModal(svg));
      diagram.style.cursor = "zoom-in";
      diagram.title = "Double-click to view in full resolution lightbox";
    });
  }

  // Observe DOM for Mermaid rendering completion
  function setupObserver() {
    const observer = new MutationObserver(() => {
      const svgs = document.querySelectorAll(".mermaid svg");
      if (svgs.length > 0) {
        attachDiagramControls();
      }
    });

    observer.observe(document.body, { childList: true, subtree: true });
    // Also try immediately
    setTimeout(attachDiagramControls, 800);
    setTimeout(attachDiagramControls, 2000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupObserver);
  } else {
    setupObserver();
  }
})();
