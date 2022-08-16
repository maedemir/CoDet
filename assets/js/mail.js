function sendMail()
{
    var name = document.getElementById('inputName').value
    var mail = document.getElementById('inputMail').value
    var phone = document.getElementById('inputPhone').value
    var type = document.getElementById('inputType').value
    var subject = document.getElementById('inputSubject').value
    var message = document.getElementById('inputMessage').value

    if(!name || !mail || !subject || !message || type === "0"){
        alert ("You should fill all parts! (Only phone number is optional)")
    }
    else{
        alert ("Mail Sent")
    }
}
