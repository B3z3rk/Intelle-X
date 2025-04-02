// <!-- JavaScript for File Upload Handling -->
    
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

      // Simulate upload action (for demonstration)
      function uploadFiles() {
        alert("Files uploaded successfully!");
      }

      document.addEventListener("DOMContentLoaded", function () {
        const bookmarkIcon = document.querySelector(".fa-bookmark");
        const floppyDiskIcon = document.querySelector(".fa-floppy-disk");
        const shareIcon = document.querySelector(".fa-share-from-square");

        // Function to save results
        function saveResults() {
          const resultsContainer = document.getElementById("results");
          const searchQuery = document.getElementById("search").value;

          if (resultsContainer) {
            localStorage.setItem("savedResults", resultsContainer.innerHTML);
            localStorage.setItem("savedQuery", searchQuery);
            alert("Results saved!");
          } else {
            alert("No results to save!");
          }
        }

        // Function to restore saved results
        function restoreResults() {
          const savedResults = localStorage.getItem("savedResults");
          const savedQuery = localStorage.getItem("savedQuery");

          if (savedResults) {
            let resultsSection = document.getElementById("results");
            if (!resultsSection) {
              resultsSection = document.createElement("div");
              resultsSection.classList.add("card");
              resultsSection.id = "results";
              document.body.appendChild(resultsSection);
            }
            resultsSection.innerHTML = onsavedResults;
            alert("Results restored!");
          } else {
            alert("No saved results found.");
          }
        }

        // Function to share results
        function shareResults() {
          const savedResults = localStorage.getItem("savedResults");
          const savedQuery = localStorage.getItem("savedQuery");

          if (savedResults) {
            const textToShare =
              `Search Results for "${savedQuery}":\n\n` +
              savedResults.replace(/<\/?[^>]+(>|$)/g, ""); // Removing HTML tags for text format
            if (navigator.share) {
              navigator
                .share({
                  title: "BM25 Search Results",
                  text: textToShare,
                  url: window.location.href,
                })
                .then(() => console.log("Shared successfully"))
                .catch((err) => console.error("Error sharing:", err));
            } else {
              alert(
                "Sharing is not supported in this browser. Copy and share manually."
              );
            }
          } else {
            alert("No results to share!");
          }
        }

        // Attach event listeners
        bookmarkIcon.addEventListener("click", saveResults);
        floppyDiskIcon.addEventListener("click", restoreResults);
        shareIcon.addEventListener("click", shareResults);
      });

      document.addEventListener("DOMContentLoaded", function() {
        const sidebar = document.getElementById("sidebar");
        const menuBtn = document.getElementById("menu-btn");
        const closeBtn = document.getElementById("close-btn");
        const content = document.getElementById("main-content");
        const body = document.body;

    
        // Open Sidebar
        menuBtn.addEventListener("click", function() {
            sidebar.classList.add("active");
            content.classList.add("shifted");
            body.classList.add('sidebar-open');
            content.style.marginLeft = '250px';
        });
    
        // Close Sidebar
        closeBtn.addEventListener("click", function() {
            sidebar.classList.remove("active");
            content.classList.remove("shifted");
            body.classList.remove('sidebar-open');

            // Delay the content shift back by the sidebar's transition duration
            setTimeout(() => {
                 content.style.marginLeft = "0";}, 500); //
        });
    
    });
    
    