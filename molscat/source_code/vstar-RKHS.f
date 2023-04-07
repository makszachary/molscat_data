      SUBROUTINE VINIT(II,RM,EPSIL)
      USE physical_constants, ONLY: bohr_to_Angstrom,hartree_in_inv_cm
      USE potential, ONLY: LAMBDA
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      SAVE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 28/05/13       STATUS: FINISHED                 |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C      THIS SUBROUTINE INCLUDES TO BOUND & MOLSCAT, VIA THE VSTAR      |
C      MECHANISM, 4 'DIABATIC' POTENTIAL CURVES FOR Li(2S)-Yb(3P)      |
C              (3{2Sigma+}, 2{2Pi}, 1{4Pi}, 1{4Sigma+})                |
C                            AS DESCRIBED IN:                          |
C                   J. CHEM. PHYS. 133, 124317 (2010)                  |
C                     DISPERSION COEFFICIENTS FROM:                    |
C       (AT INPUT, X IN ANGSTROMS.  AT OUTPUT, ENERGIES IN CM-1)       |
C-----------------------------------------------------------------------

C PARAMETERS
      PARAMETER (MAXPOT=10,mxfnln=100)
C INPUT VARIABLES
      CHARACTER(LEN=MXFNLN), DIMENSION(MAXPOT) :: FILENAMES
      INTEGER, DIMENSION(MAXPOT) :: ILAM, INUM
      double precision, DIMENSION(MAXPOT) :: xscl
      CHARACTER(LEN=3) LAMNUM
      logical csn_print,debug
C INTERNAL VARIABLES
      double precision, allocatable :: RR(:,:),VV(:,:),CSN(:,:),
     1                                 ANMK(:,:),ALPHA(:,:), RAs(:),
     2                                 ECONVS(:),RCONVS(:)
      integer, allocatable :: npts(:), ns(:), ms(:), ss(:), ncsns(:)
      double precision, allocatable :: eshfts(:), scls(:)
      logical, allocatable :: asymps(:)
      integer s
      logical asymp
C NAMELIST (NOTE THAT THIS LIST SHOULD MATCH THAT IN potenl.f)
      NAMELIST/RKHS_CONTROL/ FILENAMES, ILAM, INUM, LAMNUM, NPOT,
     1                       csn_print, xscl
C
      Debug=.False.
      IF (II.NE.1) RETURN

      WRITE(6,*) "  INITIALISING GENERAL MULTI-POTENTIAL RKHS ROUTINE"

      DO I=1,MAXPOT
        FILENAMES(I)=''
        ILAM(I)=0
        INUM(I)=0
        xscl(I)=1.0
      ENDDO
      LAMNUM="LAM"
      csn_print=.false.
      READ(5,RKHS_CONTROL)
      WRITE(6,*) "  REQUESTED ",NPOT," POTENTIALS."
      IF (LAMNUM.EQ."LAM") THEN
        WRITE(6,*) "  POTENTIALS ARE LABELLED BY VALUES OF LAMBDA, ",
     1             "WHICH WILL BE MATCHED TO VALUES SPECIFIED IN &POTL"
      ELSEIF (LAMNUM.EQ."NUM") THEN
        WRITE(6,*) "  POTENTIALS ARE LABELLED BY THE INDEX TO THE ",
     1             "LAMBDA ARRAY SPECIFIED IN &POTL"
      ELSE
        WRITE(6,*) "  INVALID VALUE OF LAMNUM = ",LAMNUM
        STOP
      ENDIF

c the following is an ugly construction, reading just the headers of the
c potential files, then coming back to them later, but we need some way to find
c out how to dimension the allocatable arrays for the potential points. The
c alternatives could be to hard-code large enough max values, input them, or
c dynamically expand the arrays as we go through the files
c At this point, we technically only need to pick out a few values from the
c headers, but it's convenient to process the first two lines fully now
      allocate(npts(npot),eshfts(npot),scls(npot),ns(npot),ms(npot),
     1         ss(npot),asymps(npot),ncsns(npot),RAs(npot),rconvs(npot),
     2         econvs(npot))
      do i = 1,npot
        open(9,file=trim(filenames(i)),status='old',err=2010)
c check err is appropriate
        read(9,*)
C Number of points given in the potential.
C Energy shift added to each point (in same units as the potential), set to 0 if not wanted.
C Scaling factor (applied after energy shift), set to 1 if not wanted.
C Units of R (given in terms of RM in &POTL), set to 1 if not wanted.
C Units of energy (given in terms of EPSIL in &POTL), set to 1 if not wanted.
C Note that scl and econv do the same thing in terms of the given potential points, but only econv is applied to the long-range coefficients
        read(9,*) npt,ESHIFT,scl,RCONV,ECONV
        read(9,*)
C Integer parameters n, m, and s, controling the RKHS.
C Whether or not you want to enforce specific long-range coefficients.
C Number of long-range coefficients given.
        read(9,*) n,m,s,asymp,ncsn
C I kinda want to add an extra unit conversion for the CSN, but it's not needed right now, so I'll leave it
        mxnpt=max(mxnpt,npt)
        mxncsn=max(mxncsn,ncsn)
        mxn=max(mxn,n)
        npts(i)=npt
        eshfts(i)=eshift
        scls(i)=scl
        rconvs(i)=rconv
        econvs(i)=econv
        ns(i)=n
        ms(i)=m
        ss(i)=s
        asymps(i)=asymp
        ncsns(i)=ncsn
        close(9)
      enddo
      mxnpt=maxval(npts)
      mxncsn=maxval(ncsns)
      mxn=maxval(ns)
c can now allocate arrays with the correct dimensions
      allocate(RR(npot,mxnpt+mxncsn),VV(npot,mxnpt),csn(npot,mxncsn),
     1         anmk(npot,mxn),alpha(npot,mxnpt+mxncsn))
c initialise all these arrays with NaN. This *shouldn't* make any difference,
c but if some indexing goes astray, it will return NaNs rather than failing
c silently. I'm sure this would horrify better coders than me!
      A=0.0
      do i=1,npot
        do j=1,mxnpt+mxncsn
          RR(i,j)=A/A
          alpha(i,j)=A/A
        enddo
        do j=1,mxnpt
          VV(i,j)=A/A
        enddo
        do j=1,mxncsn
          csn(i,j)=A/A
        enddo
        do j=1,mxn
          anmk(i,j)=A/A
        enddo
      enddo

      do i = 1,npot
        open(9,file=trim(filenames(i)),status='old',err=2010)
c check err is appropriate
        read(9,*)
        read(9,*)
        read(9,*)
        read(9,*)
        read(9,*)
C R at which the long-range terms will be calculated. Only matters for numerical stability I think? Should probably be somewhere in the mid-range.
C NCSN long-range inverse-power coefficients, starting with coefficient of  R^-s(m+1) and incrementing by s
C NCSN R values to apply the virtual points associated with the forced long-range coefficients. Should be outside the 
C All in the units as defined above
        read(9,*) RAs(i),CSN(i,1:ncsns(i)),
     1            RR(i,npts(i)+1:npts(i)+ncsns(i))
        read(9,*)
        RAS(i)=RAS(i)*RCONV
        do j=1,ncsns(i)
          ipow=s*(m+j)
          csn(i,j)=csn(i,j)*econv*rconv**ipow
          RR(i,npts(i)+j)=RR(i,npts(i)+j)*rconv
        enddo
        do j =1,npts(i)
          read(9,*) RR(i,j),VV(i,j)
c will probably include these lines for unit conversions, just not sure yet of the best way to input them
          RR(i,j)=RR(i,j)*RCONV
          VV(i,j)=(VV(i,j)-eshfts(i))*scls(i)*ECONV*xscl(i)
        enddo
        WRITE(6,9) i,scls(i),xscl(i)
 9    FORMAT(2X,"SHORT-RANGE POTENTIAL ",I1," SCALING FACTOR (POTENTIAL FILE)*(INPUT FILE) =",2X,F14.12,2X,"*",2X,F14.12) 

c on second thoughts, maybe we do want to pass explicit length slices here, rather than assumed size arrays
        call RKHS_INIT(RR(i,1:),VV(i,1:),NPTS(i),Ns(i),ms(i),ss(i),
     1                 asymps(i),csn(i,1:),ncsns(i),RAs(i),anmk(i,1:),
     2                 alpha(i,1:),CSN_print)
ccc
ccc write some generic boilerplate output for initialisation
ccc
ccc wonder if we want any option for debug output
ccc
        if(debug) then
          write(*,*)"pot ",I
          write(*,*)"npt,N,M,S ",npts(i),Ns(i),ms(i),ss(i)
          write(*,*)"RR ",RR(i,:)
          write(*,*)"VV ",VV(i,:)
          write(*,*)"asymp,ncsn,ra ",asymps(i),ncsns(i),ras(i)
          write(*,*)"csn ",csn(i,:)
          write(*,*)"anmk ",anmk(i,:)
          write(*,*)"alpha ",alpha(i,:)
        endif
      enddo
      if(debug) then
        write(*,*)"total matricies"
        write(*,*)"npt ",npts
        write(*,*)"N ",Ns
        write(*,*)"M ",ms
        write(*,*)"S ",ss
        write(*,*)"RR ",RR
        write(*,*)"VV ",VV
        write(*,*)"asymp ",asymps
        write(*,*)"ncsm ",ncsns
        write(*,*)"ra ",ras
        write(*,*)"csn ",csn
        write(*,*)"anmk ",anmk
        write(*,*)"alpha ",alpha
      endif
C
      RETURN
C-----------------------------------------------------------------------
      ENTRY VSTAR(II,X,SUM)
C-----------------------------------------------------------------------
      if (debug) then
        write(*,*) "in vstar call"
        write(*,*) "ii,x ",ii,x
        write(*,*) "RR(ii) ",RR(ii,:)
        write(*,*) "VV(ii) ",VV(ii,:)
        write(*,*) "alpha(ii) ",alpha(ii,:)
      endif
      VPOT=V_RKHS(X,RR(ii,:),NPTS(ii)+NCSNS(ii),Ns(ii),ms(ii),ss(ii),
     1            anmk(ii,:),alpha(ii,:))
      sum=vpot
      return
C DUMMY ROUTINES
      ENTRY VSTAR1(II,X,SUM)
      ENTRY VSTAR2(II,X,SUM)
      PRINT*, '* * * ERROR * * * SUBROUTINE VSTAR'
      STOP    ' VSTAR: DERIVATIVES NOT IMPLEMENTED'
C
 2010 PRINT*, " ERROR IN V32SP_INIT: COULD NOT OPEN FILE ",FILE_IN
      RETURN
      END SUBROUTINE VINIT
C*************************** RKHS_INIT *********************************
      SUBROUTINE RKHS_INIT(RR,VV,NPT,N,M,S,ASYMP,CSN,NCSN,RA,ANMK,ALPHA,
     & CSN_PRINT)
      IMPLICIT NONE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 20/06/12       STATUS: FINISHED, TESTED         |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C   THIS ROUTINE INITIALISES THE ANMK AND ALPHA ARRAYS USED BY V_RKHS  |
C            EQUATIONS FROM J. CHEM. PHYS. 113, 3960 (2000)            |
C-----------------------------------------------------------------------
C USES ANMK_EVAL (+ LAPACK'S DGESV & DCOPY)
      INTEGER NN,N,M,S,NCSN,NPT,I,J,K,IPIV,INFO
      REAL*8 RR(NPT+NCSN),VV(NPT),CSN(MAX(1,NCSN)),RA
      REAL*8 ANMK(N),ALPHA(NPT+NCSN),GAMMAS,BETA,RMIN,RMAX,CDISP
      LOGICAL ASYMP,CSN_PRINT
      ALLOCATABLE GAMMAS(:,:),BETA(:),IPIV(:)
C
      IF (S.LE.0) STOP " ERROR IN RKHS_INIT: S <= 0"
      IF (ASYMP .AND. NCSN.GT.N) STOP " ERROR IN RKHS_INIT: NCSN > N"
      NN=NPT+NCSN
      ALLOCATE (GAMMAS(NN,NN))
      ALLOCATE (BETA(NN))
      ALLOCATE (IPIV(NN))
C GETS A_{NMK=0,N) COEEFICIENTS USING EQ.(8)
      CALL ANMK_EVAL(N,M,ANMK)
      LI : DO I=1,NN ! LOOPS EVALUATE GAMMA^(S)_{IJ} FROM EQ.(9) & BETA_J FROM EQ.(10)
         LJ : DO J=1,NN
            IF (J.LE.NPT) THEN
               RMIN=MIN(RR(I),RR(J))
               RMAX=MAX(RR(I),RR(J))
               GAMMAS(J,I)=0.D0
               DO K=0,N-1
                  GAMMAS(J,I)=GAMMAS(J,I)+ANMK(K+1)*(RMIN/RMAX)**(S*K)
               ENDDO
               GAMMAS(J,I)=GAMMAS(J,I)*RMAX**(-S*(M+1)) ! Q^{N,M}_1(RI^S,RJ^S) FROM EQ.(1)
            ELSE
               K=J-NPT-1
               GAMMAS(J,I)=ANMK(K+1)*RR(I)**(S*K)/RA**(S*(K+M+1))
            ENDIF
         ENDDO LJ
         IF (I.LE.NPT) THEN
            BETA(I)=VV(I)
         ELSE
            K=I-NPT-1
            BETA(I)=-CSN(K+1)*RA**(-S*(K+M+1))
         ENDIF
      ENDDO LI
      CALL DGESV(NN,1,GAMMAS,NN,IPIV,BETA,NN,INFO) ! SOLVES EQ.(8) FOR ALPHA
      CALL DCOPY(NN,BETA,1,ALPHA,1) ! COPIES SOLUTION INTO ALPHA
      DEALLOCATE (GAMMAS,BETA,IPIV)
c (IF REQUESTED = CSN_PRINT) PRINTS OUT THE N DISPERSION COEFFICIENTS
      IF (CSN_PRINT) THEN
         WRITE(6,*) "RKHS_INIT: DISPERSION COEFFICIENTS"
         DO K=0,N-1
            CDISP=0.D0
            DO I=1,NN
               CDISP=CDISP+ALPHA(I)*RR(I)**(S*K)
            ENDDO
            CDISP=-CDISP*ANMK(K+1) ! C_{S(K+M+1)} FROM EQ.(6)
            IF (K+1.LE.NCSN) THEN
               WRITE(6,1) S*(K+M+1),CDISP,CSN(K+1),
     &                    ABS((CDISP-CSN(K+1))/(CDISP+CSN(K+1)))*200
            ELSE
               WRITE(6,2) S*(K+M+1),CDISP
            ENDIF
         ENDDO
      ENDIF
C
      RETURN
 1    FORMAT(2X,"C",I2," = ",F18.6," (out) ",F18.6," (in)",
     &       E15.2," (% error)")
 2    FORMAT(2X,"C",I2," = ",F18.6," (out) ")
      END SUBROUTINE RKHS_INIT
C**************************** V_RKHS ***********************************
      FUNCTION V_RKHS(R,RR,NN,N,M,S,ANMK,ALPHA)
      IMPLICIT NONE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 19/06/12       STATUS: FINISHED, TESTED         |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C     THIS FUNCTION INTERPOLATES/EXTRAPOLATES USING THE RKHS METHOD    |
C            EQUATIONS FROM J. CHEM. PHYS. 113, 3960 (2000)            |
C-----------------------------------------------------------------------
      INTEGER NN,N,M,S,I,K
      REAL*8 R,RR(NN),ANMK(N),ALPHA(NN),V_RKHS,RMIN,RMAX,QNM1
C
      V_RKHS=0.D0
      DO I=1,NN ! LOOPS EVALUATE Q^{N,M}_1(RI^S,R^S) FROM EQ.(1)
         RMIN=MIN(R,RR(I))
         RMAX=MAX(R,RR(I))
         QNM1=0.D0
         DO K=0,N-1
            QNM1=QNM1+ANMK(K+1)*(RMIN/RMAX)**(S*K)
         ENDDO
         QNM1=QNM1*RMAX**(-S*(M+1))
         V_RKHS=V_RKHS+QNM1*ALPHA(I)! EVALUATES V USING EQ.(4)
      ENDDO
C
      RETURN
      END FUNCTION V_RKHS
C****************************** ANMK_EVAL ******************************
      SUBROUTINE ANMK_EVAL(N,M,ANMK)
      IMPLICIT NONE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 19/06/12       STATUS: FINISHED, TESTED         |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C  THIS ROUTINE EVALUATES THE (ARRAY) OF ANMK_{N M 0:N-1) COEFFICIENTS |
C           USING EQ.(7) IN J. CHEM. PHYS. 113, 3960 (2000)            |
C [NOTICE: A_{NMK} == BETA^{N,M}_K IN J. CHEM. PHYS. 112, 4415 (2000)] |
C-----------------------------------------------------------------------
C USES FACTORIAL, POCHHAMMER (+ BETA)
      INTEGER I,J,N,M,K
      INTEGER*8 FACTORIAL,POCHHAMMER
      REAL*8 BETA,ANMK(N),FCT
C STATEMENT FUNCTIONS
C  BETA FUNCTION WITH INTEGER ARGS: BETA(I,J) = (I-1)!(J-1)!/(I+J-1)!
      BETA(I,J)=FACTORIAL(I-1)*FACTORIAL(J-1)/DFLOAT(FACTORIAL(I+J-1))
C
      FCT=N*N*BETA(N,M+1)
      DO K=0,N-1
         ANMK(K+1)=FCT*POCHHAMMER(-N+1,K)*POCHHAMMER(M+1,K)
     &             /DFLOAT(POCHHAMMER(N+M+1,K)*FACTORIAL(K))
      ENDDO
C
      RETURN
      END SUBROUTINE ANMK_EVAL
C****************************** FACTORIAL ******************************
      FUNCTION FACTORIAL(N)
      IMPLICIT NONE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 19/06/12       STATUS: FINISHED, TESTED         |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C             THIS FUNCTION EVALUATES THE FACTORIAL OF N               |
C    (IT USES A VERY SIMPLE ALGORITHM, IT SHOULD BE OK FOR SMALL N)    |
C-----------------------------------------------------------------------
      INTEGER N,FCT10(0:10),I
      INTEGER*8 FACTORIAL
      DATA FCT10/1,1,2,6,24,120,720,5040,40320,362880,3628800/
C
      SELECT CASE (N)
      CASE (:-1)
         STOP " ERROR IN FACTORIAL: N < 0"
      CASE (0:10)
         FACTORIAL=FCT10(N)
      CASE DEFAULT
         FACTORIAL=FCT10(10)
         DO I=11,N
            FACTORIAL=FACTORIAL*I
         ENDDO
      END SELECT
C
      RETURN
      END FUNCTION FACTORIAL
C****************************** POCHAMMER ******************************
      FUNCTION POCHHAMMER(I,N)
      IMPLICIT NONE
C-----------------------------------------------------------------------
C  DATE (LAST UPDATE): 19/06/12       STATUS: FINISHED, TESTED         |
C  AUTHOR: MAYKEL LEONARDO GONZALEZ MARTINEZ                           |
C-----------------------------------------------------------------------
C    THIS FUNCTION EVALUATES THE POCHHAMMER'S SYMBOL (I, N INTEGERS)   |
C            (I)N = {1 IF N = 0; I(I+1)...(I+N-1) IF N > 0             |
C  (IT USES A VERY SIMPLE ALGORITHM, IT SHOULD BE OK FOR SMALL I & N)  |
C-----------------------------------------------------------------------
      INTEGER I,J,N
      INTEGER*8 POCHHAMMER
C
      SELECT CASE (N)
      CASE (:-1)
         STOP " ERROR IN POCHHAMMER: N < 0"
      CASE (0)
         POCHHAMMER=1
      CASE DEFAULT
         POCHHAMMER=I
         DO J=1,N-1
            POCHHAMMER=POCHHAMMER*(I+J)
         ENDDO
      END SELECT
C
      RETURN
      END FUNCTION POCHHAMMER
