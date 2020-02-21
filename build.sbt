val nd4jVersion = "1.0.0-alpha"

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "kenrota",
      scalaVersion := "2.11.12", // use 2.11.x for nd4j
      version      := "0.0.1"
    )),
    name := "myonn",
    libraryDependencies += "org.nd4j" % "nd4j-api" % nd4jVersion,
    libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
    libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion,
    libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.26",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3",
  )
