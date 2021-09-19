function handleFileSelect(){
    var hr = new XMLHttpRequest();
    var url = "model.php";
    var fn = document.getElementById("upload").files[0];
    var formData = new FormData();
    formData.append("file", fn);
    hr.open("POST", url, true);
    hr.onreadystatechange = function() {
        if(hr.readyState == 4 && hr.status == 200) {
            var return_data = hr.responseText;
            alert(fn);
        }
    }
    hr.send(formData);
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

document.getElementById("upload").addEventListener("change", handleFileSelect);

