function handleFileSelect(){
    var hr = new XMLHttpRequest();
    var url = "model.php";
    var fn = document.getElementById("upload").files[0];
    var formData = new FormData();
    window.alert("Request Sent");
    formData.append("file", fn);
    hr.open("POST", url, true);
    window.alert("opened");
    hr.onreadystatechange = function() {
        var return_data = hr.responseText;
        window.alert(return_data);
        window.alert(fn);
    }
    window.alert("pre-send");
    hr.send(formData);
    window.alert("post-send");
}

// function handleFileSelect(evt){
//     evt.preventDefault();
//     const files = document.getElementById('upload').files;
//     const formData = new FormData()
//     for (let i = 0; i < files.length; i++) {
//       let file = files[i]
//       formData.append('files[]', file)
//     }
//     window.alert(files);
//     fetch(url, {
//         method: 'POST',
//         body: formData,
//     }).then((response) => {console.log(response)})
//     // var url = "./model.php?filename"
// }

