// <!-- JavaScript for application -->

const themeIcon = document.getElementById("theme");

// Apply saved theme on load
window.onload = function () {
  const theme = localStorage.getItem("theme");
  if (theme === "dark") {
    document.body.classList.add("dark-mode");
    themeIcon.classList.replace("fa-lightbulb", "fa-moon");
  }
};

themeIcon.addEventListener("click", function () {
  document.body.classList.toggle("dark-mode");
  const isDark = document.body.classList.contains("dark-mode");
  localStorage.setItem("theme", isDark ? "dark" : "light");

  // Toggle icon
  if (isDark) {
    themeIcon.classList.replace("fa-lightbulb", "fa-moon");
  } else {
    themeIcon.classList.replace("fa-moon", "fa-lightbulb");
  }
});

// Handle file selection for display
document
  .getElementById("file-input")
  .addEventListener("change", function (event) {
    const fileList = document.getElementById("file-list");
    fileList.innerHTML = ""; // Clear previous list

    const files = event.target.files;
    for (const file of files) {
      const listItem = document.createElement("li");
      listItem.textContent = file.name;
      fileList.appendChild(listItem);
    }
  });

// Allow drag-and-drop file upload
function allowDrop(event) {
  event.preventDefault();
}

function handleFileDrop(event) {
  event.preventDefault();
  const files = event.dataTransfer.files;
  const fileInput = document.getElementById("file-input");
  fileInput.files = files;

  // Display files in the file list
  const fileList = document.getElementById("file-list");
  fileList.innerHTML = ""; // Clear previous list
  for (const file of files) {
    const listItem = document.createElement("li");
    listItem.textContent = file.name;
    fileList.appendChild(listItem);
  }
}

// Handle file selection
function handleFileSelect(event) {
  const files = event.target.files;
  const fileList = document.getElementById("file-list");
  fileList.innerHTML = ""; // Clear previous list

  for (const file of files) {
    const listItem = document.createElement("li");
    listItem.textContent = file.name;
    fileList.appendChild(listItem);
  }
}

// Trigger file input click event
function triggerFileInput() {
  document.getElementById("file-input").click();
}

document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.getElementById("sidebar");
  const menuBtn = document.getElementById("menu-btn");
  const closeBtn = document.getElementById("close-btn");
  const content = document.getElementById("main-content");
  const body = document.body;

  // Open Sidebar
  menuBtn.addEventListener("click", function () {
    sidebar.classList.add("active");
    content.classList.add("shifted");
    body.classList.add("sidebar-open");
    content.style.marginLeft = "250px";
  });

  // Close Sidebar
  closeBtn.addEventListener("click", function () {
    sidebar.classList.remove("active");
    content.classList.remove("shifted");
    body.classList.remove("sidebar-open");

    setTimeout(() => {
      content.style.marginLeft = "0";
    }, 500); //
  });
});

//view document

document.addEventListener("DOMContentLoaded", function () {
  const icon = document.getElementById("view-icon");

  if (icon) {
    icon.addEventListener("click", () => {
      const filename = icon.dataset.filename;
      window.open(`/view/${filename}`, "_blank");
    });
  }
});

document.addEventListener("DOMContentLoaded", function () {
  const deleteFile = document.getElementById("trash");

  if (deleteFile) {
    deleteFile.addEventListener("click", () => {
      const checkedBoxes = document.querySelectorAll(
        'input[name="selected_files"]:checked'
      );
      const selectedFilenames = Array.from(checkedBoxes).map((cb) => cb.value);

      if (selectedFilenames.length === 0) {
        alert("Please select at least one file to delete.");
        return;
      }

      fetch("/home", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          operation: "delete",
          selected_files: selectedFilenames,
        }),
      }).then((response) => {
        if (response.ok) {
          console.log("here");
          location.reload();
        } else {
          alert("Error deleting files.");
        }
      });
    });
  }
});

//dark mode
const toggleDarkMode = () => {
  document.body.classList.toggle("dark-mode");
};
