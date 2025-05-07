// <!-- JavaScript for application -->

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
    body.classList.add('sidebar-open');
    content.style.marginLeft = '250px';
  });

  // Close Sidebar
  closeBtn.addEventListener("click", function () {
    sidebar.classList.remove("active");
    content.classList.remove("shifted");
    body.classList.remove('sidebar-open');


    setTimeout(() => {
      content.style.marginLeft = "0";
    }, 500); //
  });

});

//view document

document.addEventListener("DOMContentLoaded", function () {
  const icon = document.getElementById("view-icon");
  console.log("here")
  if (icon) {
    icon.addEventListener("click", () => {
      const filename = icon.dataset.filename;
      window.open(`/view/${filename}`, "_blank");
    });
  }
});
