var xhr;
if (window.ActiveXObject) {
    xhr = new ActiveXObject ("Microsoft.XMLHTTP");
}
else if (window.XMLHttpRequest) {
    xhr = new XMLHttpRequest();
}

const url = "model.php";

function handleFileSelect(evt){
    evt.preventDefault();
    const files = document.getElementById('upload').files;
    const file = files[0];
    window.alert(files);
    window.alert(file);
    fetch(url, {
        method: 'POST',
        body: file,
    }).then((response) => {console.log(response)})
    // var url = "./model.php?filename"
}

document.getElementById('upload').addEventListener('change', handleFileSelect, false);

