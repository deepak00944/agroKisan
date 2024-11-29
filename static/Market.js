
// event jo ho rha h button dabane prr brr brr go krne prr

document.addEventListener("DOMContentLoaded", function () {
  const firstDiv = document.getElementById("main");
  const secondDiv = document.getElementById("load");
  const result = document.getElementById("result");
  const showButton = document.getElementById("btn_predict");
  const hamburger = document.querySelector('.hamburger');
  const nav = document.querySelector('header nav');

  hamburger.addEventListener('click', () => {
    nav.classList.toggle('show');
  });

  function handleButtonClick() {
    firstDiv.style.display = "none";
    secondDiv.style.display = "block";
    result.style.display = "none";

    setTimeout(function () {
      secondDiv.style.display = "none";
      result.style.display = "block";
    }, 1000);

    setTimeout(function () {
      result.style.display = "none";
      firstDiv.style.display = "block";
    }, 1000);


    setTimeout(function () {
      result.style.display = "block";
      fourthDiv.style.display = "none";
    }, 1000);


  }

  showButton.addEventListener("click", handleButtonClick);
});


function makeFieldsRequired() {
  var form = document.querySelector('form');
  var inputElements = form.querySelectorAll('input');
  var selectElements = form.querySelectorAll('select');

  inputElements.forEach(function (input) {
    if (!input.hasAttribute('required')) {
      input.setAttribute('required', true);
    }
  });

  selectElements.forEach(function (select) {
    if (!select.hasAttribute('required')) {
      select.setAttribute('required', true);
    }
  });
}

// Call the function to make fields required
makeFieldsRequired();
