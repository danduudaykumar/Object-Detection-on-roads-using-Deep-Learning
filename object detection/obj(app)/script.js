document.getElementById("login-form").addEventListener("submit", function(event) {
    // Clear any previous error messages
    document.getElementById("username-error").textContent = '';
    document.getElementById("password-error").textContent = '';
    
    let isValid = true;

    // Get form values
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // Basic validation (can be customized)
    if (username.length < 3 || username.length > 20) {
        document.getElementById("username-error").textContent = "Username must be between 3 and 20 characters.";
        isValid = false;
    }

    if (password.length < 6) {
        document.getElementById("password-error").textContent = "Password must be at least 6 characters.";
        isValid = false;
    }

    // If validation fails, prevent form submission
    if (!isValid) {
        event.preventDefault();
    }
});
