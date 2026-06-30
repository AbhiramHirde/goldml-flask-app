/* ============================================================
   GoldML — main.js
   Global JS utilities for the Flask dashboard
   ============================================================ */

// ── Active nav highlight (fallback for Flask endpoint check) ────────────────
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname;
  document.querySelectorAll('#mainNav .nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href && href !== '#' && path === href) {
      link.classList.add('active');
    }
  });

  // ── Animate KPI numbers on scroll ──────────────────────────────────────
  const kpiValues = document.querySelectorAll('.kpi-value');
  if (kpiValues.length) {
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateNumber(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });
    kpiValues.forEach(el => observer.observe(el));
  }

  // ── Predict form loading state ──────────────────────────────────────────
  const form = document.getElementById('predictForm');
  const btn  = document.getElementById('predictBtn');
  if (form && btn) {
    form.addEventListener('submit', () => {
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Predicting…';
      btn.disabled = true;
    });
  }

  // ── Smooth scroll for anchor links ─────────────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', e => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Auto-dismiss alerts after 5s ───────────────────────────────────────
  document.querySelectorAll('.alert:not(.alert-info)').forEach(alert => {
    setTimeout(() => {
      alert.style.transition = 'opacity 0.5s';
      alert.style.opacity = '0';
      setTimeout(() => alert.remove(), 500);
    }, 5000);
  });
});

// ── Animate a numeric counter ───────────────────────────────────────────────
function animateNumber(el) {
  const raw   = el.textContent.trim();
  const isNum = /^[\d,]+(\.\d+)?%?$/.test(raw);
  if (!isNum) return;

  const hasPct  = raw.includes('%');
  const hasDot  = raw.includes('.');
  const target  = parseFloat(raw.replace(/[,%]/g, ''));
  const decimals = hasDot ? (raw.split('.')[1]?.replace('%','').length || 0) : 0;
  const start   = 0;
  const duration = 1000;
  const startTs = performance.now();

  function step(ts) {
    const progress = Math.min((ts - startTs) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);
    const current  = start + (target - start) * eased;
    el.textContent = current.toFixed(decimals) + (hasPct ? '%' : '');
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Global chart defaults ────────────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size   = 12;
  Chart.defaults.color       = '#64748b';
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15,23,42,0.9)';
  Chart.defaults.plugins.tooltip.titleColor      = '#f5a623';
  Chart.defaults.plugins.tooltip.bodyColor       = '#e2e8f0';
  Chart.defaults.plugins.tooltip.cornerRadius    = 8;
  Chart.defaults.plugins.tooltip.padding         = 10;
}
