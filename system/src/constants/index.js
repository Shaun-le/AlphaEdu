import {
    benefitIcon1,
    benefitIcon2,
    benefitIcon3,
    benefitIcon4,
    benefitImage2,
    chromecast,
    disc02,
    discord,
    discordBlack,
    facebook,
    figma,
    file02,
    framer,
    homeSmile,
    instagram,
    notification2,
    notification3,
    notification4,
    notification5,
    notion,
    photoshop,
    plusSquare,
    protopie,
    raindrop,
    recording01,
    recording03,
    roadmap1,
    roadmap2,
    roadmap3,
    roadmap4,
    searchMd,
    slack,
    sliders04,
    telegram,
    github,
    twitter,
    gmail,
    doc,
    pdf,
    csv,
    json,
    txt,
    biology,
    geography,
  } from "../assets"
  
  export const navigation = [
    {
      id: "0",
      title: "Overview",
      url: "#overview",
    },
    {
      id: "1",
      title: "Features",
      url: "#features",
    },
    {
      id: "2",
      title: "How to use",
      url: "#how-to-use",
    },
    /*{
      id: "2",
      title: "Roadmap",
      url: "#roadmap",
    },*/
    {
      id: "3",
      title: "Pricing",
      url: "#pricing",
    },
    {
      id: "4",
      title: "New account",
      url: "#signup",
      onlyMobile: true,
    },
    {
      id: "5",
      title: "Sign in",
      url: "#login",
      onlyMobile: true,
    },
  ];
  
  export const heroIcons = [homeSmile, file02, searchMd, plusSquare];
  
  export const notificationImages = [notification5];
  
  export const companyLogos = [biology,geography];
  
  export const brainwaveServices = [
    "Photo generating",
    "Photo enhance",
    "Seamless Integration",
  ];
  
  export const brainwaveServicesIcons = [
    recording03,
    recording01,
    disc02,
    chromecast,
    sliders04,
  ];
  
  export const roadmap = [
    {
      id: "0",
      title: "Voice recognition",
      text: "Enable the chatbot to understand and respond to voice commands, making it easier for users to interact with the app hands-free.",
      date: "May 2023",
      status: "done",
      imageUrl: roadmap1,
      colorful: true,
    },
    {
      id: "1",
      title: "Gamification",
      text: "Add game-like elements, such as badges or leaderboards, to incentivize users to engage with the chatbot more frequently.",
      date: "May 2023",
      status: "progress",
      imageUrl: roadmap2,
    },
    {
      id: "2",
      title: "Chatbot customization",
      text: "Allow users to customize the chatbot's appearance and behavior, making it more engaging and fun to interact with.",
      date: "May 2023",
      status: "done",
      imageUrl: roadmap3,
    },
    {
      id: "3",
      title: "Integration with APIs",
      text: "Allow the chatbot to access external data sources, such as weather APIs or news APIs, to provide more relevant recommendations.",
      date: "May 2023",
      status: "progress",
      imageUrl: roadmap4,
    },
  ];
  
  export const qS = 
    "question: Lê Thái Tổ tên thật là gì?"
  export const aS = 
    "answer: Lê Lợi"

  export const qO = "question: Khởi nghĩa Lam Sơn nổ ra năm nào?"
  export const aO = 'answer: B. 1418'
  export const cO = 'A. 1417 - B. 1418 - C. 1419 - D. 1420'

  export const qF = "question: Năm 1418, Lê Lợi tổ chức cuộc ____."
  export const aF = 'answer: A. khởi nghĩa Lam Sơn'
  
  export const collabContent = [
    {
      id: "0",
      title: "Subjective test",
      question: qS,
      answer: aS
    },
    {
      id: "1",
      title: "Objective test",
      question: qO,
      mrc: cO,
      answer: aO,
    },
    {
      id: "2",
      title: "Fill in the blank",
      question: qF,
      answer: aF
    },
  ];
  
  export const collabApps = [
    {
      id: "0",
      title: "Figma",
      icon: figma,
      width: 36,
      height: 36,
    },
    {
      id: "1",
      title: "Notion",
      icon: notion,
      width: 34,
      height: 34,
    },
    {
      id: "2",
      title: "Discord",
      icon: discord,
      width: 44,
      height: 44,
    },
    {
      id: "3",
      title: "Slack",
      icon: slack,
      width: 40,
      height: 40,
    },
    {
      id: "4",
      title: "Photoshop",
      icon: photoshop,
      width: 40,
      height: 40,
    },
    /*{
      id: "5",
      title: "Protopie",
      icon: protopie,
      width: 34,
      height: 34,
    },
    {
      id: "6",
      title: "Framer",
      icon: framer,
      width: 26,
      height: 34,
    },
    {
      id: "7",
      title: "Raindrop",
      icon: raindrop,
      width: 38,
      height: 32,
    },*/
  ];

  export const fileTypes = [
    {
      id: "0",
      title: "Doc",
      icon: doc,
      width: 44,
      height: 44,
    },
    {
      id: "1",
      title: "PDF",
      icon: pdf,
      width: 30,
      height: 30,
    },
    {
      id: "2",
      title: "TXT",
      icon: txt,
      width: 32,
      height: 32,
    },
    {
      id: "3",
      title: "CSV",
      icon: csv,
      width: 38,
      height: 38,
    },
    {
      id: "4",
      title: "JSON",
      icon: json,
      width: 38,
      height: 38,
    }
  ]
  
  export const pricing = [
    {
      id: "0",
      title: "Basic",
      description: "AI chatbot, personalized recommendations",
      price: "0",
      features: [
        "An AI chatbot that can understand your queries",
        "Personalized recommendations based on your preferences",
        "Ability to explore the app and its features without any cost",
      ],
    },
    {
      id: "1",
      title: "Premium",
      description: "Advanced AI chatbot, priority support, analytics dashboard",
      price: "9.99",
      features: [
        "An advanced AI chatbot that can understand complex queries",
        "An analytics dashboard to track your conversations",
        "Priority support to solve issues quickly",
      ],
    },
    {
      id: "2",
      title: "Enterprise",
      description: "Custom AI chatbot, advanced analytics, dedicated account",
      price: null,
      features: [
        "An AI chatbot that can understand your queries",
        "Personalized recommendations based on your preferences",
        "Ability to explore the app and its features without any cost",
      ],
    },
  ];
  
  export const benefits = [
    {
      id: "0",
      title: "Ask anything",
      text: "Lets users quickly find answers to their questions without having to search through multiple sources.",
      backgroundUrl: "assets/benefits/card-1.svg",
      iconUrl: benefitIcon1,
      imageUrl: benefitImage2,
    },
    {
      id: "1",
      title: "Improve everyday",
      text: "The app uses natural language processing to understand user queries and provide accurate and relevant responses.",
      backgroundUrl: "assets/benefits/card-2.svg",
      iconUrl: benefitIcon2,
      imageUrl: benefitImage2,
      light: true,
    },
    {
      id: "2",
      title: "Connect everywhere",
      text: "Connect with the AI chatbot from anywhere, on any device, making it more accessible and convenient.",
      backgroundUrl: "assets/benefits/card-3.svg",
      iconUrl: benefitIcon3,
      imageUrl: benefitImage2,
    },
    {
      id: "3",
      title: "Fast responding",
      text: "Lets users quickly find answers to their questions without having to search through multiple sources.",
      backgroundUrl: "assets/benefits/card-4.svg",
      iconUrl: benefitIcon4,
      imageUrl: benefitImage2,
      light: true,
    },
    {
      id: "4",
      title: "Ask anything",
      text: "Lets users quickly find answers to their questions without having to search through multiple sources.",
      backgroundUrl: "assets/benefits/card-5.svg",
      iconUrl: benefitIcon1,
      imageUrl: benefitImage2,
    },
    {
      id: "5",
      title: "Improve everyday",
      text: "The app uses natural language processing to understand user queries and provide accurate and relevant responses.",
      backgroundUrl: "assets/benefits/card-6.svg",
      iconUrl: benefitIcon2,
      imageUrl: benefitImage2,
    },
  ];
  
  export const socials = [
    {
      id: "0",
      title: "Github",
      iconUrl: github,
      url: "https://github.com/Shaun-le",
    },
    {
      id: "1",
      title: "Gmail",
      iconUrl: gmail,
      url: "mailto:lehuuloi.cs@gmail.com",
    },
    {
      id: "2",
      title: "Instagram",
      iconUrl: instagram,
      url: "https://www.instagram.com/shnaunee/",
    },
    /*{
      id: "3",
      title: "Telegram",
      iconUrl: telegram,
      url: "#",
    },*/
    {
      id: "3",
      title: "Facebook",
      iconUrl: facebook,
      url: "https://www.facebook.com/shnaunee/",
    },
];