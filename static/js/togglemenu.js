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
    document.getElementById("analize_button").style.display = "inline";
};

function Method() {
    document.getElementById("methodology").style.display = "block";
    document.getElementById("some_results").style.display = "none";
    document.getElementById("method_a").setAttribute("class", "clicked");
    document.getElementById("result_a").setAttribute("class", "not_clicked");
};

function Result() {
    document.getElementById("methodology").style.display = "none";
    document.getElementById("some_results").style.display = "block";
    document.getElementById("method_a").setAttribute("class", "not_clicked");
    document.getElementById("result_a").setAttribute("class", "clicked");
};