# Step 1 aur Step 7 ke documents

## 1. Sir ko message (Step 1)

Seedha copy karke email / WhatsApp kar dein. Jawab aane par neeche "Jawab" wale hisse mein likh kar
commit kar dein (roadmap: "Done jab jawab notes mein likhe hon").

> Assalam o Alaikum Sir,
>
> PiLENS ke v2 par kaam shuru ho gaya hai. v1 (ResNet18 + LSTM, 2 classes) baseline ke taur par freeze
> hai, aur v2 mein X3D-S ka two-stage cascade hai (pehle binary anomaly detection, alert par 14-class
> UCF-Crime), jo Raspberry Pi 5 par ONNX Runtime ke saath chalega. Abhi tak Kaggle par binary stage ka
> video AUC ~0.91 aur 14-class accuracy ~33% (chance 7%) aaya hai. Pi par latency, throughput aur
> temperature measure karna agla step hai.
>
> Paper shuru karne se pehle chaar baaton par aapki rehnumai chahiye:
>
> 1. **Papers ki tadaad:** Kya ek hi paper hoga (system + model, v1 vs v2 ablation ke saath), ya system
>    aur model ke alag do papers?
> 2. **Venue aur template:** Target IEEE Access soch raha hoon. Kya ye theek hai, ya koi aur
>    journal/conference behtar hogi?
> 3. **Authorship order:** Authors aur unka order kya hoga?
> 4. **Ethics / consent:** Night-vision ke liye hum apne day/night clips record karna chahte hain
>    (team ke log, written consent ke saath). Department ki ethics approval ka kya tareeqa hai, aur
>    kya v1 ke WhatsApp videos paper mein use ho sakte hain?
>
> Shukriya,
> Huzaifa Asad

### Jawab (sir se baat ke baad bharein)

| Sawal | Jawab | Tareekh |
|---|---|---|
| 1. Ek paper ya do | | |
| 2. Venue / template | | |
| 3. Authorship order | | |
| 4. Ethics / consent | | |

---

## 2. Recording consent form (Step 7)

Har us shakhs se sign karwayein jo apni recorded clips mein nazar aaye. Department ka apna form ho to
wahi use karein; ye sirf template hai, legal advice nahi.

**PiLENS: Video Recording Consent Form**

Project: PiLENS: real-time suspicious activity detection on Raspberry Pi (research project)
Researcher(s): ______________________  Supervisor: ______________________
Department / University: ______________________

1. **Purpose.** I understand that I will be recorded on video, by day and at night (infrared camera),
   while acting out normal and staged "suspicious" activities (e.g. fighting, theft) for training and
   testing a computer-vision system.
2. **Staged only.** All suspicious activities are acted. No real harm, weapon or crime is involved,
   and I may refuse any action I am not comfortable with.
3. **Use of the data.** The recordings will be used to train and evaluate the system and may appear
   in research publications as (tick one):
   - [ ] frames with faces blurred only
   - [ ] unblurred frames
   - [ ] not shown in any publication (used for training/testing only)
4. **Storage.** Recordings are stored on access-restricted university / project storage, are not
   shared publicly, and will be deleted by ____ / ____ / ______ unless I agree otherwise.
5. **Withdrawal.** Participation is voluntary. I can withdraw at any time before
   ____ / ____ / ______ without giving a reason, and my recordings will then be deleted.
6. **Contact.** Questions: ______________________ (email / phone).

Name: ______________________  Signature: ______________________  Date: ____ / ____ / ______

Researcher signature: ______________________  Date: ____ / ____ / ______

### Recording checklist

- [ ] Har participant ka signed form (scan karke private folder mein; repo mein **kabhi nahi**)
- [ ] Ethics approval reference (agar department maange): ______________________
- [ ] Day aur night dono, wahi camera + fixed exposure/gain jo deployment mein hai
- [ ] Har clip ke liye event start/end frame aur Day/Night `annotations.csv` mein (same columns)
- [ ] Night test set alag rakha (training mein nahi)
- [ ] Recordings repo mein commit nahi hongi (privacy); sirf annotation CSV aur split lists
