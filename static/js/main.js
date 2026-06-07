"use strict";

document.addEventListener("DOMContentLoaded", () => {
  initFormValidation();
  initNumericFormatting();
  initConfirmDialogs();
});

function initFormValidation() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    if (!form.checkValidity()) {
      e.preventDefault();
      e.stopPropagation();
      highlightEmptyFields(form);
    }
    form.classList.add("was-validated");
  });

  form.querySelectorAll("input[type=number]").forEach(input => {
    input.addEventListener("blur", () => validateNumericField(input));
  });
}

function highlightEmptyFields(form) {
  form.querySelectorAll("[required]").forEach(el => {
    if (!el.value) {
      el.classList.add("is-invalid");
      el.addEventListener("change", () => el.classList.remove("is-invalid"), { once: true });
    }
  });
}

function validateNumericField(input) {
  const val = parseFloat(input.value);
  const min = input.hasAttribute("min") ? parseFloat(input.getAttribute("min")) : null;

  if (isNaN(val)) {
    setFieldError(input, "Please enter a valid number.");
    return false;
  }
  if (min !== null && val < min) {
    setFieldError(input, `Value must be at least ${min}.`);
    return false;
  }
  clearFieldError(input);
  return true;
}

function setFieldError(input, msg) {
  input.classList.add("is-invalid");
  let fb = input.parentElement.querySelector(".invalid-feedback");
  if (!fb) {
    fb = document.createElement("div");
    fb.className = "invalid-feedback";
    input.parentElement.appendChild(fb);
  }
  fb.textContent = msg;
}

function clearFieldError(input) {
  input.classList.remove("is-invalid");
  const fb = input.parentElement.querySelector(".invalid-feedback");
  if (fb) fb.textContent = "";
}

function initNumericFormatting() {
  document.querySelectorAll("input[type=number]").forEach(input => {
    input.addEventListener("wheel", e => e.preventDefault());
    input.addEventListener("keydown", e => {
      if (e.key === "e" || e.key === "E" || e.key === "+" || e.key === "-") {
        e.preventDefault();
      }
    });
  });
}

function initConfirmDialogs() {
  document.querySelectorAll("[data-confirm]").forEach(el => {
    el.addEventListener("click", function (e) {
      if (!confirm(this.dataset.confirm)) e.preventDefault();
    });
  });
}

function deleteRecord(pid) {
  if (!confirm("Delete this prediction record?")) return;

  fetch(`/api/history/${pid}`, { method: "DELETE" })
    .then(r => r.json())
    .then(() => {
      const row = document.querySelector(`tr[data-pid="${pid}"]`);
      if (row) row.remove();
    })
    .catch(() => alert("Failed to delete record."));
}
