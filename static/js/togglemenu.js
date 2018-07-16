function ToggleNav() {
    var x = document.getElementById("circle_nav");
    if (x.style.display === "block") {
        x.style.display = "none";
    } else {
        x.style.display = "block";
    };
};

function UploadToggle() {
    document.getElementById('upload_button_hidden').click();
    var y = document.getElementById("analize_button");
    y.style.display = "inline";
};
