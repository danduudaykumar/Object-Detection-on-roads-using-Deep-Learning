document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("login-form").addEventListener("submit", function (event) {
        // Clear any previous error messages
        document.getElementById("username-error").textContent = '';
        document.getElementById("password-error").textContent = '';

        let isValid = true;

        // Get form values and trim whitespace
        const username = document.getElementById("username").value.trim();
        const password = document.getElementById("password").value.trim();

        // Username validation: alphanumeric, 3-20 characters
        const usernameRegex = /^[a-zA-Z0-9]{3,20}$/;
        if (!usernameRegex.test(username)) {
            document.getElementById("username-error").textContent = "Username must be 3-20 alphanumeric characters.";
            isValid = false;
        }

        // Password validation: at least 6 characters, 1 uppercase, 1 lowercase, 1 number
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{6,}$/;
        if (!passwordRegex.test(password)) {
            document.getElementById("password-error").textContent = 
                "Password must be at least 6 characters and include 1 uppercase, 1 lowercase, and 1 number.";
            isValid = false;
        }

        // If validation fails, prevent form submission
        if (!isValid) {
            event.preventDefault();
        }
    });
});
