   // firebase config
   var config = {
    apiKey: "AIzaSyCFus5s9f9JSygMC9l6oKgy2NndG5SFwHc",
    authDomain: "time4break-b3569.firebaseapp.com",
    projectId: "time4break-b3569",
    storageBucket: "time4break-b3569.appspot.com",
    messagingSenderId: "184415951680",
    appId: "1:184415951680:web:75504dfc735192f6b0f8a5",
    measurementId: "G-Q2N70XDRK7",
    databaseURL: "https://time4break-b3569-default-rtdb.firebaseio.com/",
  };
  
  let userId = null;
  

  function initializeFirebase() {
    firebase.initializeApp(config);

    /////////////////////////////////////
  
    /**********************\
       * Check login status *
      \**********************/
    var database = firebase.database();
    var isStart = false;
    // save the user's profile into Firebase so we can list users,
    // use them in Security and Firebase Rules, and show profiles
    function writeUserData(userId, name, email, imageUrl) {
      firebase
        .database()
        .ref("users/" + userId)
        .set({
          username: name,
          email: email,
          //some more user data
        });
    }
    // let userId = null;
    firebase.auth().onAuthStateChanged(function (user) {
      if (user) {
        userId = user.uid;
        getSuggestions(userId);
      } else {
        // Set the contents of the div with the id suggestions to say "Log in to view past suggestions"
        document.getElementById("suggestions").innerHTML =
          "Log in to view past suggestions";
      }
    });
  }

  setInterval(() => {
    console.log("IN INTERVAL");
    if (userId) {
      getSuggestions(userId);
      getState();
      getCurrentSuggestion();
    }
  }, 10000);

  var suggestions = [];

  function startVideo() {
    isStart = !isStart;
    document.getElementById("startButton").innerHTML = isStart
      ? "Stop"
      : "Start";
    firebase
      .database()
      .ref("/startState/")
      .set({
        startState: isStart ? "start" : "stop",
      });
  }
  function getState() {
    var starCountRef = firebase.database().ref("state");
    starCountRef.on("value", (snapshot) => {
      const data = snapshot.val();
      if (data) {
        document.getElementById("current_state").innerHTML =
          "Current State: " + data;
      }
    });
  }
  async function getCurrentSuggestion() {
    var starCountRef = firebase.database().ref("currentSuggestion");
    await getSuggestions(userId);
    starCountRef.on("value", (snapshot) => {
      const data = snapshot.val();
      if (data) {
        document.getElementById("current_sugg").innerHTML =
          "Current Suggestion: " + data;
      }
      if (suggestions.includes(data)) {
        console.log("Already Exists");
      } else {
        firebase
          .database()
          .ref("users/" + userId)
          .set({
            suggestions: [...suggestions, data],
            //some more user data
          });
      }
    });
    getSuggestions(userId);
  }
  async function getSuggestions(userId) {
    firebase
      .database()
      .ref("users/" + userId)
      .once("value")
      .then(function (snapshot) {
        suggestions = snapshot.val().suggestions;
        $("#suggestions").empty();
        var list = document.getElementById("suggestions");
        for (var i = 0; i < suggestions.length; i++) {
          var item = document.createElement("li");
          item.innerHTML = suggestions[i];
          list.appendChild(item);
        }
      });
  }